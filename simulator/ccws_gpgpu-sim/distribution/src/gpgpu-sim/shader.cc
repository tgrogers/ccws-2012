// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// George L. Yuan, Andrew Turner, Inderpreet Singh 
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// Neither the name of The University of British Columbia nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <float.h>
#include "shader.h"
#include "gpu-sim.h"
#include "addrdec.h"
#include "dram.h"
#include "stat-tool.h"
#include "gpu-misc.h"
#include "../cuda-sim/ptx_sim.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/cuda-sim.h"
#include "gpu-sim.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "visualizer.h"
#include "../intersim/statwraper.h"
#include "../intersim/interconnect_interface.h"
#include "icnt_wrapper.h"
#include "../print_streams.h"
#include "dynamic_per_pc_stats.h"
#include <string.h>
#include <limits.h>
#include "scheduling-point-system.h"

#define PRIORITIZE_MSHR_OVER_WB 1
#define MAX(a,b) (((a)>(b))?(a):(b))

#define ENABLE_MEM_DIV_PRINT 0
#define ENABLE_MEM_DIV_STATS 0

extern unsigned L1_HIT_LATENCY;
extern unsigned AMMOUNT_ADDED_BY_THREAD;
extern unsigned g_ptx_sim_num_insn;

bool is_cc_scheduler( unsigned type ) {
    return  CACHE_CONSCIOUS_WARP_SCHEDULER == type || REAL_CACHE_CONSCIOUS_WARP_SCHEDULER == type ||
            REAL_MAB_CACHE_CONSCIOUS_WARP_SCHEDULER == type;
}

/////////////////////////////////////////////////////////////////////////////

std::list<unsigned> shader_core_ctx::get_regs_written( const inst_t &fvt ) const
{
   std::list<unsigned> result;
   for( unsigned op=0; op < MAX_REG_OPERANDS; op++ ) {
      int reg_num = fvt.arch_reg.dst[op]; // this math needs to match that used in function_info::ptx_decode_inst
      if( reg_num >= 0 ) // valid register
         result.push_back(reg_num);
   }
   return result;
}

shader_core_ctx::shader_core_ctx( class gpgpu_sim *gpu, 
                                  class simt_core_cluster *cluster,
                                  unsigned shader_id,
                                  unsigned tpc_id,
                                  const struct shader_core_config *config,
                                  const struct memory_config *mem_config,
                                  shader_core_stats *stats )
   : m_barriers( config->max_warps_per_shader, config->max_cta_per_core ), m_dynamic_warp_id(0), m_scheduling_point_system(config->max_warps_per_shader, &config->m_point_system_config, stats, m_warp, shader_id)
{
   m_kernel = NULL;
   m_gpu = gpu;
   m_cluster = cluster;
   m_config = config;
   m_memory_config = mem_config;
   m_stats = stats;
   unsigned warp_size=config->warp_size;

   m_sid = shader_id;
   m_tpc = tpc_id;

   m_pipeline_reg.reserve(N_PIPELINE_STAGES);
   for (int j = 0; j<N_PIPELINE_STAGES; j++) {
      m_pipeline_reg.push_back(register_set(m_config->pipe_widths[j],pipeline_stage_name_decode[j]));
   }

   m_threadState = (thread_ctx_t*) calloc(sizeof(thread_ctx_t), config->n_thread_per_shader);
   m_thread = (ptx_thread_info**) calloc(sizeof(ptx_thread_info*), config->n_thread_per_shader);

   m_not_completed = 0;
   m_active_threads.reset();
   m_n_active_cta = 0;
   for (unsigned i = 0; i<MAX_CTA_PER_SHADER; i++  ) 
      m_cta_status[i]=0;
   for (unsigned i = 0; i<config->n_thread_per_shader; i++) {
      m_thread[i]= NULL;
      m_threadState[i].m_cta_id = -1;
      m_threadState[i].m_active = false;
   }
   
   // m_icnt = new shader_memory_interface(this,cluster);
    if ( m_config->gpgpu_perfect_mem ) {
        m_icnt = new perfect_memory_interface(this,cluster);
    } else {
        m_icnt = new shader_memory_interface(this,cluster);
    }
   m_mem_fetch_allocator = new shader_core_mem_fetch_allocator(shader_id,tpc_id,mem_config);

   // fetch
   m_last_warp_fetched = -1;

   #define STRSIZE 1024
   char name[STRSIZE];
   snprintf(name, STRSIZE, "L1I_%03d", m_sid);
   m_L1I = new read_only_cache( name,m_config->m_L1I_config,m_sid,get_shader_instruction_cache_id(),m_icnt,IN_L1I_MISS_QUEUE, false);

   initilizeSIMTStack(config->max_warps_per_shader,this->get_config()->warp_size);
   m_scoreboard = new Scoreboard(m_sid, m_config->max_warps_per_shader);

   if ( m_config->m_simplified_config.enabled ) {
      m_ldst_unit = new simplified_mem_system_ldst_unit( m_icnt, m_mem_fetch_allocator, this, &m_operand_collector, m_scoreboard, config, mem_config, stats, shader_id, tpc_id, m_gpu->get_simple_mem_sys() );
   } else if ( is_cc_scheduler( m_config->m_warp_scheduler_config.m_scheduler_type ) ) {
      m_ldst_unit = new cache_conscious_ldst_unit( m_icnt, m_mem_fetch_allocator, this, &m_operand_collector, m_scoreboard, config, mem_config, stats, shader_id, tpc_id );
   } else if (POINT_SYSTEM_WARP_SCHEDULER == m_config->m_warp_scheduler_config.m_scheduler_type) {
       m_ldst_unit = new point_system_ldst_unit( m_icnt, m_mem_fetch_allocator, this, &m_operand_collector, m_scoreboard, config, mem_config, stats, shader_id, tpc_id, &m_scheduling_point_system );
   } else {
      m_ldst_unit = new ldst_unit( m_icnt, m_mem_fetch_allocator, this, &m_operand_collector, m_scoreboard, config, mem_config, stats, shader_id, tpc_id );
   }

   m_warp.resize( m_config->max_warps_per_shader );
   if ( REAL_MAB_CACHE_CONSCIOUS_WARP_SCHEDULER == m_config->m_warp_scheduler_config.m_scheduler_type ) {
       for ( unsigned i = 0; i < m_config->max_warps_per_shader; ++i ) {
           m_warp[ i ] = new mem_allow_bit_shd_warp_t(this,
                   warp_size,
                   dynamic_cast< cache_conscious_ldst_unit* >( m_ldst_unit ) );
       }
   } else {
       for ( unsigned i = 0; i < m_config->max_warps_per_shader; ++i ) {
           m_warp[ i ] = new shd_warp_t(this, warp_size);
       }
   }

   //op collector configuration
   enum { SP_CUS, SFU_CUS, MEM_CUS, GEN_CUS };
   m_operand_collector.add_cu_set(SP_CUS, m_config->gpgpu_operand_collector_num_units_sp, m_config->gpgpu_operand_collector_num_out_ports_sp);
   m_operand_collector.add_cu_set(SFU_CUS, m_config->gpgpu_operand_collector_num_units_sfu, m_config->gpgpu_operand_collector_num_out_ports_sfu);
   m_operand_collector.add_cu_set(MEM_CUS, m_config->gpgpu_operand_collector_num_units_mem, m_config->gpgpu_operand_collector_num_out_ports_mem);
   m_operand_collector.add_cu_set(GEN_CUS, m_config->gpgpu_operand_collector_num_units_gen, m_config->gpgpu_operand_collector_num_out_ports_gen);

   opndcoll_rfu_t::port_vector_t in_ports;
   opndcoll_rfu_t::port_vector_t out_ports;
   opndcoll_rfu_t::uint_vector_t cu_sets;
   for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sp; i++) {
       in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
       out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
       cu_sets.push_back((unsigned)SP_CUS);
       cu_sets.push_back((unsigned)GEN_CUS);
       m_operand_collector.add_port(in_ports,out_ports,cu_sets);
       in_ports.clear(),out_ports.clear(),cu_sets.clear();
   }

   for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_sfu; i++) {
       in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
       out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
       cu_sets.push_back((unsigned)SFU_CUS);
       cu_sets.push_back((unsigned)GEN_CUS);
       m_operand_collector.add_port(in_ports,out_ports,cu_sets);
       in_ports.clear(),out_ports.clear(),cu_sets.clear();
   }

   for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_mem; i++) {
       in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);
       out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);
       cu_sets.push_back((unsigned)MEM_CUS);
       cu_sets.push_back((unsigned)GEN_CUS);                       
       m_operand_collector.add_port(in_ports,out_ports,cu_sets);
       in_ports.clear(),out_ports.clear(),cu_sets.clear();
   }   


   for (unsigned i = 0; i < m_config->gpgpu_operand_collector_num_in_ports_gen; i++) {
       in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
       in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
       in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);
       out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
       out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
       out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);
       cu_sets.push_back((unsigned)GEN_CUS);   
       m_operand_collector.add_port(in_ports,out_ports,cu_sets);
       in_ports.clear(),out_ports.clear(),cu_sets.clear();
   }

   m_operand_collector.init( m_config->gpgpu_num_reg_banks, this );

   // execute
   m_num_function_units = m_config->gpgpu_num_sp_units + m_config->gpgpu_num_sfu_units + 1; // sp_unit, sfu, ldst_unit
   //m_dispatch_port = new enum pipeline_stage_name_t[ m_num_function_units ];
   //m_issue_port = new enum pipeline_stage_name_t[ m_num_function_units ];

   //m_fu = new simd_function_unit*[m_num_function_units];

   for (int k = 0; k < m_config->gpgpu_num_sp_units; k++) {
       m_fu.push_back(new sp_unit( &m_pipeline_reg[EX_WB], m_config ));
       m_dispatch_port.push_back(ID_OC_SP);
       m_issue_port.push_back(OC_EX_SP);
   }

   for (int k = 0; k < m_config->gpgpu_num_sfu_units; k++) {
       m_fu.push_back(new sfu( &m_pipeline_reg[EX_WB], m_config ));
       m_dispatch_port.push_back(ID_OC_SFU);
       m_issue_port.push_back(OC_EX_SFU);
   }



   m_fu.push_back(m_ldst_unit);
   m_dispatch_port.push_back(ID_OC_MEM);
   m_issue_port.push_back(OC_EX_MEM);

   assert(m_num_function_units == m_fu.size() and m_fu.size() == m_dispatch_port.size() and m_fu.size() == m_issue_port.size());

   //there are as many result buses as the width of the EX_WB stage
   num_result_bus = config->pipe_widths[EX_WB];
   for(int i=0; i<num_result_bus; i++){
	   this->m_result_bus.push_back(new std::bitset<MAX_ALU_LATENCY>());
   }

   m_last_inst_gpu_sim_cycle = 0;
   m_last_inst_gpu_tot_sim_cycle = 0;

    //schedulers
    //must currently occur after all inputs have been initialized.
    for (int i = 0; i < m_config->gpgpu_num_sched_per_core; i++) {
        if ( LOOSE_ROUND_ROBIN_WARP_SCHEDULER == m_config->m_warp_scheduler_config.m_scheduler_type ) {
            schedulers.push_back( new loose_round_robin_scheduler_unit(m_stats,this,m_scoreboard,m_simt_stack,&m_warp,
                &m_pipeline_reg[ID_OC_SP],
                &m_pipeline_reg[ID_OC_SFU],
                &m_pipeline_reg[ID_OC_MEM]) );
        } else if ( GREEDY_TOP_WARP_SCHEDULER == m_config->m_warp_scheduler_config.m_scheduler_type ) {
            schedulers.push_back( new greepy_top_scheduler_unit(m_stats,this,m_scoreboard,m_simt_stack,&m_warp,
                &m_pipeline_reg[ID_OC_SP],
                &m_pipeline_reg[ID_OC_SFU],
                &m_pipeline_reg[ID_OC_MEM]) );
        } else if ( CACHE_CONSCIOUS_WARP_SCHEDULER == m_config->m_warp_scheduler_config.m_scheduler_type ) {
            schedulers.push_back( new cache_conscious_scheduler_unit(m_stats,this,m_scoreboard,m_simt_stack,&m_warp,
                &m_pipeline_reg[ID_OC_SP],
                &m_pipeline_reg[ID_OC_SFU],
                &m_pipeline_reg[ID_OC_MEM],
                dynamic_cast< cache_conscious_ldst_unit* >( m_ldst_unit ), m_threadState,
                m_config->m_warp_scheduler_config.m_workless_cycles_threshold ) );
        } else if ( CUSTOM_MEMCACHED_WARP_SCHEDULER == m_config->m_warp_scheduler_config.m_scheduler_type ) {
            schedulers.push_back( new custom_memcached_scheduler_unit(m_stats,this,m_scoreboard,m_simt_stack,&m_warp,
                &m_pipeline_reg[ID_OC_SP],
                &m_pipeline_reg[ID_OC_SFU],
                &m_pipeline_reg[ID_OC_MEM],
                m_config->m_L1C_config.get_line_sz() * m_config->m_L1C_config.get_num_lines() ) );
        } else if ( GREEDY_UNTIL_STALL_WARP_SCHEDULER == m_config->m_warp_scheduler_config.m_scheduler_type ) {
            schedulers.push_back( new greedy_until_stall_scheduler_unit(m_stats,this,m_scoreboard,m_simt_stack,&m_warp,
                &m_pipeline_reg[ID_OC_SP],
                &m_pipeline_reg[ID_OC_SFU],
                &m_pipeline_reg[ID_OC_MEM]) );
        } else if ( NVIDIA_ISCA_2011_TWO_LEVEL_WARP_SCHEDULER == m_config->m_warp_scheduler_config.m_scheduler_type ) {
            schedulers.push_back( new nvidia_isca_2011_two_level_warp_scheduler(m_stats,this,m_scoreboard,m_simt_stack,&m_warp,
                &m_pipeline_reg[ID_OC_SP],
                &m_pipeline_reg[ID_OC_SFU],
                &m_pipeline_reg[ID_OC_MEM],
                m_warp.size(),
                m_config->m_warp_scheduler_config.m_active_warps,
                m_config->m_warp_scheduler_config.m_scheduling_policy ) );
        } else if ( GREEDY_THEN_OLDEST_WARP_SCHEDULER == m_config->m_warp_scheduler_config.m_scheduler_type ) {
            schedulers.push_back( new greedy_then_oldest_scheduler_unit(m_stats,this,m_scoreboard,m_simt_stack,&m_warp,
                &m_pipeline_reg[ID_OC_SP],
                &m_pipeline_reg[ID_OC_SFU],
                &m_pipeline_reg[ID_OC_MEM],
                m_warp.size() ) );
        } else if ( CONCURRENCY_LIMITED_SCHEDULER == m_config->m_warp_scheduler_config.m_scheduler_type ) {
            unsigned active_warps = 0;
            unsigned active_ctas = 0;
            if ( m_config->m_warp_scheduler_config.m_active_warps > 0 ) {
                assert( 0 == m_config->m_warp_scheduler_config.m_active_ctas ); // Specify only one of active_warps or active_ctas
                active_warps = m_config->m_warp_scheduler_config.m_active_warps;
            } else if ( m_config->m_warp_scheduler_config.m_active_ctas > 0 ) {
                assert( 0 == m_config->m_warp_scheduler_config.m_active_warps ); // Specify only one of active_warps or active_ctas
                active_ctas = m_config->m_warp_scheduler_config.m_active_ctas;
                printf( "CONCURRENCY_LIMITED_SCHEDULER_active_warps=%u\n", active_warps );
            } else {
                fprintf( stderr,
                        "CONCURRENCY_LIMITED_SCHEDULER. Something must be active! Specify only one of active_warps or active_ctas\n" );
                abort();
            }
            schedulers.push_back( new concurrency_limited_scheduler_unit(m_stats,this,m_scoreboard,m_simt_stack,&m_warp,
                &m_pipeline_reg[ID_OC_SP],
                &m_pipeline_reg[ID_OC_SFU],
                &m_pipeline_reg[ID_OC_MEM],
                active_warps,
                active_ctas,
                m_warp.size() ) );
        } else if ( TEXAS_TECH_REPORT_SCHEDULER == m_config->m_warp_scheduler_config.m_scheduler_type ) {
            schedulers.push_back( new texas_tech_report_scheduler_unit(m_stats,this,m_scoreboard,m_simt_stack,&m_warp,
                &m_pipeline_reg[ID_OC_SP],
                &m_pipeline_reg[ID_OC_SFU],
                &m_pipeline_reg[ID_OC_MEM],
                m_config->m_warp_scheduler_config.m_active_warps ) );
        } else if ( REAL_CACHE_CONSCIOUS_WARP_SCHEDULER == m_config->m_warp_scheduler_config.m_scheduler_type ) {
            schedulers.push_back( new realistic_cache_conscious_scheduler_unit(m_stats,this,m_scoreboard,m_simt_stack,&m_warp,
                &m_pipeline_reg[ID_OC_SP],
                &m_pipeline_reg[ID_OC_SFU],
                &m_pipeline_reg[ID_OC_MEM],
                dynamic_cast< cache_conscious_ldst_unit* >( m_ldst_unit ), m_threadState,
                m_config->m_warp_scheduler_config.m_workless_cycles_threshold,
                m_config->m_warp_scheduler_config.m_max_l1_probes_per_cycle,
                warp_scheduling_policy( m_config->m_warp_scheduler_config.m_scheduling_policy ) ) );
        } else if ( REAL_MAB_CACHE_CONSCIOUS_WARP_SCHEDULER == m_config->m_warp_scheduler_config.m_scheduler_type ) {
            schedulers.push_back( new realistic_memory_allow_bit_cc_scheduler_unit(m_stats,this,m_scoreboard,m_simt_stack,&m_warp,
                &m_pipeline_reg[ID_OC_SP],
                &m_pipeline_reg[ID_OC_SFU],
                &m_pipeline_reg[ID_OC_MEM],
                dynamic_cast< cache_conscious_ldst_unit* >( m_ldst_unit ), m_threadState,
                m_config->m_warp_scheduler_config.m_workless_cycles_threshold,
                m_config->m_warp_scheduler_config.m_max_l1_probes_per_cycle,
                warp_scheduling_policy( m_config->m_warp_scheduler_config.m_scheduling_policy ) ) );
        } else if ( STRICT_ROUND_ROBIN_WARP_SCHEDULER == m_config->m_warp_scheduler_config.m_scheduler_type ) {
            schedulers.push_back( new strict_round_robin_scheduler_unit(m_stats,this,m_scoreboard,m_simt_stack,&m_warp,
                           &m_pipeline_reg[ID_OC_SP],
                           &m_pipeline_reg[ID_OC_SFU],
                           &m_pipeline_reg[ID_OC_MEM]) );
        } else if (POINT_SYSTEM_WARP_SCHEDULER == m_config->m_warp_scheduler_config.m_scheduler_type ) {
            schedulers.push_back( new point_system_scheduler_unit(m_stats,this,m_scoreboard,m_simt_stack,&m_warp,
                           &m_pipeline_reg[ID_OC_SP],
                           &m_pipeline_reg[ID_OC_SFU],
                           &m_pipeline_reg[ID_OC_MEM],
                           &m_scheduling_point_system ));

        } else {
            fprintf( stderr, "Unknown Warp Scheduler\n" );
            abort();
        }
    }

    for (unsigned i = 0; i < m_warp.size(); i++) {
        //distribute i's evenly though schedulers;
        schedulers[i%m_config->gpgpu_num_sched_per_core]->add_supervised_warp_id(i);
    }
}

void shader_core_ctx::reinit(unsigned start_thread, unsigned end_thread, bool reset_not_completed ) 
{
   if( reset_not_completed ) {
       m_not_completed = 0;
       m_active_threads.reset();
   }
   for (unsigned i = start_thread; i<end_thread; i++) {
      m_threadState[i].n_insn = 0;
      m_threadState[i].m_cta_id = -1;
   }
   for (unsigned i = start_thread / m_config->warp_size; i < end_thread / m_config->warp_size; ++i) {
      m_warp[i]->reset();
      m_simt_stack[i]->reset();
   }
}

void shader_core_ctx::init_warps( unsigned cta_id, unsigned start_thread, unsigned end_thread )
{
    address_type start_pc = next_pc(start_thread);
    if (m_config->model == POST_DOMINATOR) {
        unsigned start_warp = start_thread / m_config->warp_size;
        unsigned end_warp = end_thread / m_config->warp_size + ((end_thread % m_config->warp_size)? 1 : 0);
        for (unsigned i = start_warp; i < end_warp; ++i) {
            unsigned n_active=0;
            simt_mask_t active_threads;
            for (unsigned t = 0; t < m_config->warp_size; t++) {
                unsigned hwtid = i * m_config->warp_size + t;
                if ( hwtid < end_thread ) {
                    n_active++;
                    assert( !m_active_threads.test(hwtid) );
                    m_active_threads.set( hwtid );
                    active_threads.set(t);
                }
            }
            m_simt_stack[i]->launch(start_pc,active_threads);
            m_warp[i]->init(start_pc,cta_id,i,active_threads,m_dynamic_warp_id);
            ++m_dynamic_warp_id;
            m_not_completed += n_active;
      }
   }
}

// return the next pc of a thread 
address_type shader_core_ctx::next_pc( int tid ) const
{
    if( tid == -1 ) 
        return -1;
    ptx_thread_info *the_thread = m_thread[tid];
    if ( the_thread == NULL )
        return -1;
    return the_thread->get_pc(); // PC should already be updatd to next PC at this point (was set in shader_decode() last time thread ran)
}

void gpgpu_sim::get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc )
{
    unsigned cluster_id = m_shader_config->sid_to_cluster(sid);
    m_cluster[cluster_id]->get_pdom_stack_top_info(sid,tid,pc,rpc);
}

void shader_core_ctx::get_pdom_stack_top_info( unsigned tid, unsigned *pc, unsigned *rpc ) const
{
    unsigned warp_id = tid/m_config->warp_size;
    m_simt_stack[warp_id]->get_pdom_stack_top_info(pc,rpc);
}

std::map< address_type, sourceline_and_hit_num >* simt_core_cluster::get_pointer_load_info( int core, int thread_id )
{
   if ( m_core[ core ] && m_core[ core ]->get_thread_state( thread_id )  ) {
      return &m_core[ core ]->get_thread_state( thread_id )->m_pointer_load_PCs;
   } else {
      return NULL;
   }
}

std::list< bin_with_redundant_count >* simt_core_cluster::get_bin_redundant_count( int core, int thread_id )
{
   if ( m_core[ core ] && m_core[ core ]->get_thread_state( thread_id )  ) {
      return &m_core[ core ]->get_thread_state( thread_id )->m_access_bins;
   } else {
      return NULL;
   }
}

unsigned long long simt_core_cluster::get_num_dynamic_load_info( int core, int thread_id ) {
   return m_core[ core ]->get_thread_state( thread_id )->m_num_dynmaic_loads;
}

unsigned long long simt_core_cluster::get_num_dynamic_pointer_load_info( int core, int thread_id ) {
   return m_core[ core ]->get_thread_state( thread_id )->m_num_dynmaic_pointer_loads;
}

bool cmp_list_size(const std::pair< new_addr_type, block_access_stats > &lhs,
        const std::pair< new_addr_type, block_access_stats > &rhs) {
    return lhs.second.warps_hitting_line.size() > rhs.second.warps_hitting_line.size();
}

void shader_core_stats::record_cache_block_hit( new_addr_type block_addr,
        unsigned sid,
        unsigned warp_id,
        unsigned pc,
        const active_mask_t& warp_mask ) {
    if ( m_block_address_stats[ sid ].find( block_addr ) == m_block_address_stats[ sid ].end() ) {
        // Block is not in the statlist
        block_access_stats new_stats;
        new_stats.warps_hitting_line[ warp_id ] = 1;
        new_stats.pcs_hitting_line[ pc ] = 1;
        new_stats.pc_exclusive_hits[ pc ] = 1;
        new_stats.pc_to_hit_times[ pc ].push_back( gpu_sim_cycle + gpu_tot_sim_cycle );
        new_stats.threads_hitting_line.resize( m_config->max_warps_per_shader * m_config->warp_size, 0 );
        for ( unsigned i = 0; i < m_config->warp_size; ++i ) {
            if ( warp_mask.test( i ) ) {
                new_stats.threads_hitting_line[ warp_id * m_config->warp_size + i ] = 1;
            }
        }
        new_stats.warp_to_hit_times[ warp_id ].push_back( gpu_sim_cycle + gpu_tot_sim_cycle );
        m_block_address_stats[ sid ][ block_addr ] = new_stats;
    } else {
        // Block is in the statlist, need to see if this warp/pc is in here
        if ( m_block_address_stats[ sid ][ block_addr ].warps_hitting_line.find( warp_id )
                == m_block_address_stats[ sid ][ block_addr ].warps_hitting_line.end() ) {
            // It is not in the list yet, add it
            m_block_address_stats[ sid ][ block_addr ].warps_hitting_line[ warp_id ] = 1;
        } else {
            // It is already in here, increment the hit count
            m_block_address_stats[ sid ][ block_addr ].warps_hitting_line[ warp_id ]++;
        }

        for ( unsigned i = 0; i < m_config->warp_size; ++i ) {
            if ( warp_mask.test( i ) ) {
                m_block_address_stats[ sid ][ block_addr ].threads_hitting_line[ warp_id * m_config->warp_size + i ]++;
            }
        }

        if ( m_block_address_stats[ sid ][ block_addr ].pcs_hitting_line.find( pc )
                == m_block_address_stats[ sid ][ block_addr ].pcs_hitting_line.end() ) {
            // It is not in the list yet, add it
            m_block_address_stats[ sid ][ block_addr ].pcs_hitting_line[ pc ] = 1;
        } else {
            // It is already in here, increment the hit count
            m_block_address_stats[ sid ][ block_addr ].pcs_hitting_line[ pc ]++;
        }

        if ( m_block_address_stats[ sid ][ block_addr ].pc_exclusive_hits.find( pc )
                != m_block_address_stats[ sid ][ block_addr ].pc_exclusive_hits.end() ) {
            m_block_address_stats[ sid ][ block_addr ].pc_exclusive_hits[ pc ]++;
        }

        m_block_address_stats[ sid ][ block_addr ].pc_to_hit_times[ pc ].push_back( gpu_sim_cycle + gpu_tot_sim_cycle );
        m_block_address_stats[ sid ][ block_addr ].warp_to_hit_times[ warp_id ].push_back( gpu_sim_cycle + gpu_tot_sim_cycle );
    }
}

void shader_core_stats::print_useful_byte_information() const {
    printf("Mem Divergence PC Status Dump.  Num warps = %zu\n", warp_to_pc_to_load_counters_map.size());
    printf("-------------------------------------------------------------------------------------------\n");
    std::map < int, std::map< new_addr_type, mem_div_load_counters > >::const_iterator it;
    for ( it = warp_to_pc_to_load_counters_map.begin(); it != warp_to_pc_to_load_counters_map.end(); it++ )
    {
        printf("PCs for warp %d\n", (*it).first);
        std::map< new_addr_type, mem_div_load_counters >::const_iterator it2;
        for ( it2 = (*it).second.begin(); it2 != (*it).second.end(); it2++ )
        {
            mem_div_load_counters counters = (*it2).second;
            float percent_all = (float)counters.num_times_all_bytes_used_or_eventually_used / (float)counters.num_accesses;
            float percent_3_4 = (float)counters.num_times_gte_3_quarters_bytes_used_or_eventually_used / (float)counters.num_accesses;
            float percent_1_2 = (float)counters.num_times_gte_half_bytes_used_or_eventually_used / (float)counters.num_accesses;
            float percent_1_4 = (float)counters.num_times_gte_1_quarter_used_or_eventually_used / (float)counters.num_accesses;

            printf( "PC = 0x%llx Total Accesses - %llu : All useful -  %llu (%4f) : 3/4 - %llu (%4f) : 1/2 - %llu (%4f) : 1/4 - %llu (%4f)\n"
                , (*it2).first
                , counters.num_accesses
                , counters.num_times_all_bytes_used_or_eventually_used
                , percent_all
                , counters.num_times_gte_3_quarters_bytes_used_or_eventually_used
                , percent_3_4
                , counters.num_times_gte_half_bytes_used_or_eventually_used
                , percent_1_2
                , counters.num_times_gte_1_quarter_used_or_eventually_used
                , percent_1_4 );
        }
    }
    printf("-------------------------------------------------------------------------------------------\n");
    printf( "All warps PC, Total Accesses, All useful, 3/4, 1/2, 1/4\n");
    std::map< new_addr_type, mem_div_load_counters >::const_iterator it2;
    for ( it2 = all_warps_pc_to_load_counters_map.begin(); it2 != all_warps_pc_to_load_counters_map.end(); it2++ )
    {
        mem_div_load_counters counters = (*it2).second;

        printf( "0x%llx, %llu, %llu, %llu, %llu, %llu\n"
            , (*it2).first
            , counters.num_accesses
            , counters.num_times_all_bytes_used_or_eventually_used
            , counters.num_times_gte_3_quarters_bytes_used_or_eventually_used
            , counters.num_times_gte_half_bytes_used_or_eventually_used
            , counters.num_times_gte_1_quarter_used_or_eventually_used );
    }

    printf("\n\n---------------------------------------------------------------------------------------\n");

    unsigned tot_bytes_used_now = 0;
    unsigned tot_bytes_used_later = 0;
    unsigned tot_already_bytes_used_now = 0;
    unsigned tot_already_bytes_used_later = 0;
    unsigned tot_bytes_never_used = 0;
    unsigned tot_bytes_read = 0;
    unsigned tot_unique_bytes_read = 0;

    for(unsigned i=0; i<m_div_info.size(); i++){
        printf("Bin %u: num_mem_access: %u, total_bytes: %u, total_unique_bytes: %u, useful_later_bytes: %u, already_useful_later_bytes: %u, never_used: %u cycle_start: %u,"
        "cycle_end: %llu\n", i, m_div_info.at(i).num_mem_access, m_div_info.at(i).total_bytes, m_div_info.at(i).total_unique_bytes, m_div_info.at(i).useful_later_bytes,
        m_div_info.at(i).already_useful_later_bytes, m_div_info.at(i).never_used_bytes,m_div_info.at(i).cycle_start,
        ( m_div_info.at(i).cycle_end==0 ? (gpu_tot_sim_cycle + gpu_sim_cycle) : m_div_info.at(i).cycle_end )  );

        printf("  Useful_bytes:\n");
        for(int j=0; j<8; j++){
            printf("    First use - %d/8: %d,  Used again - %d/8: %d \n", j, m_div_info.at(i).useful_bytes[j], j, m_div_info.at(i).already_useful_bytes[j]);
            tot_bytes_used_now += m_div_info.at(i).useful_bytes[j];
            tot_already_bytes_used_now += m_div_info.at(i).already_useful_bytes[j];
        }
        tot_bytes_used_later += m_div_info.at(i).useful_later_bytes;
        tot_already_bytes_used_later += m_div_info.at(i).already_useful_later_bytes;
        tot_bytes_read += m_div_info.at(i).total_bytes;
        tot_bytes_never_used += m_div_info.at(i).never_used_bytes;
        tot_unique_bytes_read += m_div_info.at(i).total_unique_bytes;
    }

    // Overall results
    printf("\nTotal bytes read: %u\nTotal unique bytes read: %u\nTotal bytes used now: %u\nTotal bytes already used now: %u\nTotal bytes used later: %u\nTotal bytes already used later: %u\n"
    "Total bytes never used: %u\n", tot_bytes_read, tot_unique_bytes_read, tot_bytes_used_now, tot_already_bytes_used_now,
    tot_bytes_used_later, tot_already_bytes_used_later, tot_bytes_never_used);

    /*
    const div_map_t *entry = NULL;
    std::map <new_addr_type,div_map_t>::const_iterator i=mem_div_map.begin();
    for(;i != mem_div_map.end(); i++){
    entry = &i->second;
    if(entry->num_access > 1)
    printf("Address %x called %u times\n", i->first, entry->num_access);
    }
    */
    printf("Format: bin, total_unique_bytes_read, unique_bytes_used_now, unique_bytes_used_later, unique_never_used, total_bytes_read, already_used_now,  already_useful_later_bytes, useful_later_bytes_read_this_cycle, cycle_start, cycle_end\n");
    /*
    for(int i=0; i<m_div_info.size(); i++){
    unsigned un = 0;
    unsigned aun = 0;
    for(int j=0; j<8; j++){
    un += m_div_info.at(i).useful_bytes[j];
    aun += m_div_info.at(i).already_useful_bytes[j];
    }

    printf("%d, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u\n", i,  m_div_info.at(i).total_unique_bytes,  un,
    m_div_info.at(i).useful_later_bytes, m_div_info.at(i).never_used_bytes, m_div_info.at(i).total_bytes, aun,
    m_div_info.at(i).already_useful_later_bytes, m_div_info.at(i).useful_later_bytes_read_now, m_div_info.at(i).cycle_start,
    ( m_div_info.at(i).cycle_end==0 ? (gpu_tot_sim_cycle + gpu_sim_cycle) : m_div_info.at(i).cycle_end )  );
    }
    */
    unsigned used_now[8] = {0};
    unsigned already_used_now[8] = {0};
    tot_bytes_used_later = 0;
    tot_already_bytes_used_later = 0;
    tot_bytes_never_used = 0;
    tot_bytes_read = 0;
    tot_unique_bytes_read = 0;
    unsigned tot_used_now = 0;
    unsigned tot_already_used_now = 0;
    for(unsigned i=0; i<m_div_info.size(); i++){
        tot_bytes_used_later += m_div_info.at(i).useful_later_bytes;
        tot_already_bytes_used_later += m_div_info.at(i).already_useful_later_bytes;
        tot_bytes_never_used += m_div_info.at(i).never_used_bytes;
        tot_bytes_read += m_div_info.at(i).total_bytes;
        tot_unique_bytes_read += m_div_info.at(i).total_unique_bytes;
        for(int j=0; j<8; j++){
            used_now[j] += m_div_info.at(i).useful_bytes[j];
            tot_used_now += m_div_info.at(i).useful_bytes[j];
            already_used_now[j] += m_div_info.at(i).already_useful_bytes[j];
            tot_already_used_now += m_div_info.at(i).already_useful_bytes[j];
        }
    }
    printf("Total unique read: %u\nTotal read: %d\nTotal bytes used later: %u\nTotal bytes used again, already marked used later: %u\n"
    "Total unique bytes never used: %u\n", tot_unique_bytes_read, tot_bytes_read, tot_bytes_used_later, tot_already_bytes_used_later, tot_bytes_never_used);
    for(int i=0; i<8; i++){
        printf("  Total used now (%d/8): %u     Total used again, already used now (%d/8): %u\n", i, used_now[i], i, already_used_now[i]);
    }
    printf("Percentages:\n");
    for(int i=0; i<8; i++){
        printf("  Total %% used now (%d/8): %.2lf %%        Total %% used again, already used now (%d/8): %.2lf %%\n", i, ((float)used_now[i]/(float)tot_used_now)*100.0, i, ((float)already_used_now[i]/(float)tot_already_used_now)*100.0);
    }


    printf("\n\n===================\n\n");
    tot_used_now = 0;

    printf("Total bytes read:       %d\n", t_bytes_read);
    printf("Total memory access:    %u\nFraction_useful_per_memory_request:\n", t_mem_access);
    for(int i=0; i<8; i++)
        tot_used_now += t_bytes_used[i];
    for(int i=0; i<8; i++)
        printf("%.2lf,  ", ((float)t_bytes_used[i]/(float)tot_used_now)*100.0);

    printf("\nBytes read:\n");
    for(int i=0; i<8; i++)
        printf("%u, ", t_bytes_used[i]);

    printf("\n\n===================\n\n");


    printf("\n\n---------------------------------------------------------------------------------------\n");
}

void shader_core_stats::print( FILE* fout ) const
{
    unsigned icount_uarch=0;
    for(unsigned i=0; i < m_config->num_shader(); i++) {
        icount_uarch += m_num_sim_insn[i];
    }
    fprintf(fout,"gpgpu_n_tot_icount = %u\n", icount_uarch);
    fprintf(fout,"gpgpu_n_stall_shd_mem = %d\n", gpgpu_n_stall_shd_mem );
    fprintf(fout,"gpgpu_n_mem_read_local = %d\n", gpgpu_n_mem_read_local);
    fprintf(fout,"gpgpu_n_mem_write_local = %d\n", gpgpu_n_mem_write_local);
    fprintf(fout,"gpgpu_n_mem_read_global = %d\n", gpgpu_n_mem_read_global);
    fprintf(fout,"gpgpu_n_mem_write_global = %d\n", gpgpu_n_mem_write_global);
    fprintf(fout,"gpgpu_n_mem_texture = %d\n", gpgpu_n_mem_texture);
    fprintf(fout,"gpgpu_n_mem_const = %d\n", gpgpu_n_mem_const);
/*
   unsigned a,m;
   for (unsigned i=0, a=0, m=0;i<m_n_shader;i++)
      m_sc[i]->L1cache_print(stdout,a,m);
   printf("L1 Data Cache Total Miss Rate = %0.3f\n", (float)m/a);
   for (i=0,a=0,m=0;i<m_n_shader;i++)
       m_sc[i]->L1texcache_print(stdout,a,m);
   printf("L1 Texture Cache Total Miss Rate = %0.3f\n", (float)m/a);
   for (i=0,a=0,m=0;i<m_n_shader;i++)
       m_sc[i]->L1constcache_print(stdout,a,m);
   printf("L1 Const Cache Total Miss Rate = %0.3f\n", (float)m/a);
*/
    fprintf(fout, "gpgpu_n_load_insn  = %d\n", gpgpu_n_load_insn);
    fprintf(fout, "gpgpu_n_store_insn = %d\n", gpgpu_n_store_insn);
    fprintf(fout, "gpgpu_n_shmem_insn = %d\n", gpgpu_n_shmem_insn);
    fprintf(fout, "gpgpu_n_tex_insn = %d\n", gpgpu_n_tex_insn);
    fprintf(fout, "gpgpu_n_const_mem_insn = %d\n", gpgpu_n_const_insn);
    fprintf(fout, "gpgpu_n_param_mem_insn = %d\n", gpgpu_n_param_insn);

    fprintf(fout, "gpgpu_n_shmem_bkconflict = %d\n", gpgpu_n_shmem_bkconflict);
    fprintf(fout, "gpgpu_n_cache_bkconflict = %d\n", gpgpu_n_cache_bkconflict);

    fprintf(fout, "gpgpu_n_intrawarp_mshr_merge = %d\n", gpgpu_n_intrawarp_mshr_merge);
    fprintf(fout, "gpgpu_n_cmem_portconflict = %d\n", gpgpu_n_cmem_portconflict);

    fprintf(fout, "gpgpu_stall_shd_mem[c_mem][bk_conf] = %d\n", gpu_stall_shd_mem_breakdown[C_MEM][BK_CONF]);
    fprintf(fout, "gpgpu_stall_shd_mem[c_mem][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[C_MEM][MSHR_RC_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[c_mem][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[C_MEM][ICNT_RC_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[t_mem][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[T_MEM][MSHR_RC_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[t_mem][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[T_MEM][ICNT_RC_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[s_mem][bk_conf] = %d\n", gpu_stall_shd_mem_breakdown[S_MEM][BK_CONF]);
    fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][bk_conf] = %d\n",
        gpu_stall_shd_mem_breakdown[G_MEM_LD][BK_CONF] +
        gpu_stall_shd_mem_breakdown[G_MEM_ST][BK_CONF] +
        gpu_stall_shd_mem_breakdown[L_MEM_LD][BK_CONF] +
        gpu_stall_shd_mem_breakdown[L_MEM_ST][BK_CONF]
        ); // coalescing stall at data cache
    fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][coal_stall] = %d\n",
        gpu_stall_shd_mem_breakdown[G_MEM_LD][COAL_STALL] +
        gpu_stall_shd_mem_breakdown[G_MEM_ST][COAL_STALL] +
        gpu_stall_shd_mem_breakdown[L_MEM_LD][COAL_STALL] +
        gpu_stall_shd_mem_breakdown[L_MEM_ST][COAL_STALL]
        ); // coalescing stall + bank conflict at data cache
    fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_LD][MSHR_RC_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_LD][ICNT_RC_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][wb_icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_LD][WB_ICNT_RC_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[g_mem_ld][wb_rsrv_fail] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_LD][WB_CACHE_RSRV_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_ST][MSHR_RC_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_ST][ICNT_RC_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][wb_icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_ST][WB_ICNT_RC_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[g_mem_st][wb_rsrv_fail] = %d\n", gpu_stall_shd_mem_breakdown[G_MEM_ST][WB_CACHE_RSRV_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_LD][MSHR_RC_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_LD][ICNT_RC_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_LD][WB_ICNT_RC_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_rsrv_fail] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_LD][WB_CACHE_RSRV_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[l_mem_st][mshr_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_ST][MSHR_RC_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[l_mem_st][icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_ST][ICNT_RC_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_icnt_rc] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_ST][WB_ICNT_RC_FAIL]);
    fprintf(fout, "gpgpu_stall_shd_mem[l_mem_ld][wb_rsrv_fail] = %d\n", gpu_stall_shd_mem_breakdown[L_MEM_ST][WB_CACHE_RSRV_FAIL]);

    fprintf(fout, "gpu_reg_bank_conflict_stalls = %d\n", gpu_reg_bank_conflict_stalls);

    fprintf(fout, "Warp Occupancy Distribution:\n");
    fprintf(fout, "Stall:%d\t", shader_cycle_distro[2]);
    fprintf(fout, "W0_Idle:%d\t", shader_cycle_distro[0]);
    fprintf(fout, "W0_Scoreboard:%d", shader_cycle_distro[1]);
    for (unsigned i = 3; i < m_config->warp_size + 3; i++)
    fprintf(fout, "\tW%d:%d", i-2, shader_cycle_distro[i]);
    fprintf(fout, "\n");

    printf("old_tot_useful_bytes_read = %llu\n", old_tot_useful_bytes_read);
    printf("old_tot_bytes_read = %llu\n", old_tot_bytes_read);
    printf("old_percentage_useful_bytes_read = %4f\n", old_tot_useful_bytes_read/(old_tot_bytes_read*1.0));

    printf("tot_useful_bytes_read = %llu\n", tot_useful_bytes_read);
    printf("tot_evetually_useful_bytes_read = %llu\n", tot_eventually_useful_bytes_read);
    printf("tot_already_evetually_useful_bytes_read = %llu\n", tot_already_eventually_useful_bytes_read);
    printf("tot_already_useful_bytes_read = %llu\n", tot_already_useful_bytes_read);
    printf("tot_mem_div_hit_bytes = %llu\n", mem_div_hit_bytes);
    printf("tot_mem_div_access_num_bytes_requested = %llu\n", mem_div_access_num_bytes_requested);
    printf( "fraction_divergent_accesses = %4f\n", mem_div_divergent_accesses / mem_div_accesses );

    // HACK STATS
    extern unsigned long long g_hack_num_xactions_gen;
    extern unsigned long long g_hack_num_warp_accesses_processed;
    printf("g_hack_num_xactions_gen = %llu\n", g_hack_num_xactions_gen);
    printf("g_hack_num_warp_accesses_processed = %llu\n", g_hack_num_warp_accesses_processed);
    printf( "avg_mem_transactions_gen_per_mem_op = %4f\n", (double)g_hack_num_xactions_gen / (double)g_hack_num_warp_accesses_processed );

    printf("tot_bytes_read = %llu\n", tot_bytes_read);
    printf("percentage_useful_bytes_read = %4f\n", tot_useful_bytes_read/(tot_bytes_read*1.0));
    printf("percentage_already_useful_bytes_read = %4f\n", tot_already_useful_bytes_read/(tot_bytes_read*1.0));
    printf("percentage_eventually_useful_bytes_read = %4f\n", tot_eventually_useful_bytes_read/(tot_bytes_read*1.0));
    printf("percentage_already_eventually_useful_bytes_read = %4f\n", tot_already_eventually_useful_bytes_read/(tot_bytes_read*1.0));

    m_print_streams->print( print_streams::WARP_SHARED_BLOCKS, "shared_block_stats:\n" );
    unsigned count = 0;
    const unsigned MAX_SHARERS = 512;
    unsigned blocks_with_index_sharers[ MAX_SHARERS + 1 ]; // Currently the zero space is unused, keeping this way for clarity
    memset( blocks_with_index_sharers, 0x0, sizeof( unsigned ) * ( MAX_SHARERS + 1 ) );
    for ( std::vector< std::map< new_addr_type, block_access_stats > >::const_iterator it = m_block_address_stats.begin();
        it != m_block_address_stats.end(); ++it, ++count ) {
        m_print_streams->print( print_streams::WARP_SHARED_BLOCKS, "sid=%d\n", count );
        // We now want to sort all the block addresses for this shader based on how many warps touch the line.
        std::vector< std::pair< new_addr_type, block_access_stats > >sorted_vector( it->begin(), it->end() );
        std::sort( sorted_vector.begin(), sorted_vector.end(), cmp_list_size );
        for ( std::vector< std::pair< new_addr_type, block_access_stats > >::const_iterator it2 = sorted_vector.begin();
                it2 != sorted_vector.end(); ++it2 ) {
            m_print_streams->print( print_streams::WARP_SHARED_BLOCKS, "block_addr=%llx: %zu total sharers\nwarps= ", it2->first, it2->second.warps_hitting_line.size() );
            assert( it2->second.warps_hitting_line.size() <= MAX_SHARERS );
            blocks_with_index_sharers[ it2->second.warps_hitting_line.size() ]++;
            for (std::map< unsigned, unsigned >::const_iterator it3 = it2->second.warps_hitting_line.begin();
                it3 != it2->second.warps_hitting_line.end(); ++it3  ) {
                m_print_streams->print( print_streams::WARP_SHARED_BLOCKS, "%u:%u, ", it3->first, it3->second );
            }
            m_print_streams->print( print_streams::WARP_SHARED_BLOCKS, "\n" );
            for (std::map< unsigned, unsigned >::const_iterator it3 = it2->second.pcs_hitting_line.begin();
                    it3 != it2->second.pcs_hitting_line.end(); ++it3  ) {
                const char* filename = NULL;
                unsigned line_num = 0;
                get_ptx_source_info( it3->first, filename, line_num );
                m_print_streams->print( print_streams::WARP_SHARED_BLOCKS, "\nsource_file=%s;source_line=%u:hits=%d, ",
                filename, line_num, it3->second );
            }
            m_print_streams->print( print_streams::WARP_SHARED_BLOCKS, "\n" );
        }
    }

   printf( "aggregate_shared_block_stats:" );
   unsigned total_blocks = 0;
   unsigned weighted_avg = 0;
   unsigned total_shared_blocks = 0;
   unsigned weighted_shared_avg = 0;
   for ( unsigned i = 0; i < MAX_SHARERS + 1; ++i ) {
       printf( "%u, ", blocks_with_index_sharers[ i ] );
       weighted_avg += i * blocks_with_index_sharers[ i ];
       total_blocks+=blocks_with_index_sharers[ i ];
       if ( i > 1 ) {
           total_shared_blocks += blocks_with_index_sharers[ i ];
           weighted_shared_avg += i * blocks_with_index_sharers[ i ];
       }
   }
   printf( "\ntotal_hit_blocks=%u\n", total_blocks );
   printf( "total_shared_blocks=%u\n", total_shared_blocks );
   printf( "average_sharers_per_block=%f\n", (float)( weighted_avg )/(float)( total_blocks ) );
   printf( "average_sharers_per_shared_block=%f\n", (float)( weighted_shared_avg )/(float)( total_shared_blocks ) );
   printf( "percent_blocks_shared=%f\n", (float)total_shared_blocks/(float)total_blocks );

   count = 0;
   unsigned hits_total_over_all_cores = 0;
   unsigned insertions_total_over_all_cores = 0;
   for ( std::vector< unsigned >::const_iterator
           iter = m_num_shared_only_cache_hits.begin(), iter2 = m_num_shared_only_cache_insertions.begin();
           iter != m_num_shared_only_cache_hits.end(); ++iter, ++count, ++iter2 ) {
       printf( "m_num_shared_only_cache_hits_%02u=%u\n", count, *iter );
       printf( "m_num_shared_only_cache_intertions_%02u=%u\n", count, *iter2 );
       hits_total_over_all_cores += *iter;
       insertions_total_over_all_cores += *iter2;
   }
   printf( "total_num_shared_only_cache_hits=%u\n", hits_total_over_all_cores );
   printf( "total_num_shared_only_cache_insertions=%u\n", insertions_total_over_all_cores );

   if ( m_config->gpgpu_mem_visualizer_enabled ) {
       print_memory_visualization();
   }

   if ( m_print_streams->is_enabled( print_streams::USEFUL_BYTE_INFO ) ) {
       print_useful_byte_information();
   }
}

void shader_core_stats::visualizer_print( gzFile visualizer_file )
{
    // warp divergence breakdown
    gzprintf(visualizer_file, "WarpDivergenceBreakdown:");
    unsigned int total=0;
    unsigned int cf = (m_config->gpgpu_warpdistro_shader==-1)?m_config->num_shader():1;
    gzprintf(visualizer_file, " %d", (shader_cycle_distro[0] - last_shader_cycle_distro[0]) / cf );
    gzprintf(visualizer_file, " %d", (shader_cycle_distro[1] - last_shader_cycle_distro[1]) / cf );
    gzprintf(visualizer_file, " %d", (shader_cycle_distro[2] - last_shader_cycle_distro[2]) / cf );
    for (unsigned i=0; i<m_config->warp_size+3; i++) {
       if ( i>=3 ) {
          total += (shader_cycle_distro[i] - last_shader_cycle_distro[i]);
          if ( ((i-3) % (m_config->warp_size/8)) == ((m_config->warp_size/8)-1) ) {
             gzprintf(visualizer_file, " %d", total / cf );
             total=0;
          }
       }
       last_shader_cycle_distro[i] = shader_cycle_distro[i];
    }
    gzprintf(visualizer_file,"\n");

    gzprintf(visualizer_file, "WarpIssueBreakdown:");
    unsigned sid = m_config->gpgpu_warp_issue_shader;

    unsigned total_warps_assigned = m_max_thread_assigned / m_config->warp_size;
    if ( m_max_thread_assigned % m_config->warp_size != 0 ) {
        ++total_warps_assigned;
    }
    for ( unsigned i = 0; i <= total_warps_assigned; i++ ) {
    	gzprintf( visualizer_file, " %d", ( m_shader_warp_issue_distro[ sid ][ i ] - m_last_shader_warp_issue_distro[ sid ][ i ] ) );
        m_last_shader_warp_issue_distro[ sid ][ i ] = m_shader_warp_issue_distro[ sid ][ i ];
    }
    gzprintf(visualizer_file,"\n");

    gzprintf(visualizer_file, "DL1CacheBreakdown:");
    sid = m_config->gpgpu_warp_cache_distro_shader;
    // Print these in reverse order so hits are on the bottom
    for ( int i = NUM_CACHE_REQUEST_STATUS - 1; i >= 0; --i ) {
        gzprintf( visualizer_file,
                    " %d", ( m_shader_l1_data_cache_distro[ sid ][ i ] - m_last_shader_l1_data_cache_distro[ sid ][ i ] ) );
        m_last_shader_l1_data_cache_distro[ sid ][ i ] = m_shader_l1_data_cache_distro[ sid ][ i ];
    }
    gzprintf(visualizer_file,"\n");

    // Print these in reverse order so hits are on the bottom
    for ( unsigned warp_id = 0; warp_id < m_config->max_warps_per_shader; ++warp_id ) {
        gzprintf(visualizer_file, "DL1CacheS0PerW%dBreakdown:", warp_id);
        for ( int i = NUM_CACHE_REQUEST_STATUS - 1; i >= 0; --i ) {
            gzprintf( visualizer_file,
                        " %d", ( m_shader_dl1_s0_per_warp_distro[ warp_id ][ i ] - m_last_shader_dl1_s0_per_warp_distro[ warp_id ][ i ] ) );
            m_last_shader_dl1_s0_per_warp_distro[ warp_id ][ i ] = m_shader_dl1_s0_per_warp_distro[ warp_id ][ i ];
    }
        gzprintf(visualizer_file,"\n");
    }

    // overall cache miss rates
    gzprintf(visualizer_file, "gpgpu_n_cache_bkconflict: %d\n", gpgpu_n_cache_bkconflict);
    gzprintf(visualizer_file, "gpgpu_n_shmem_bkconflict: %d\n", gpgpu_n_shmem_bkconflict);     

   // instruction count per shader core
   gzprintf(visualizer_file, "shaderinsncount:  ");
   for (unsigned i=0;i<m_config->num_shader();i++) 
      gzprintf(visualizer_file, "%u ", m_num_sim_insn[i] );
   gzprintf(visualizer_file, "\n");
   // warp instruction count per shader core
   gzprintf(visualizer_file, "shaderwarpinsncount:  ");
   for (unsigned i=0;i<m_config->num_shader();i++) 
      gzprintf(visualizer_file, "%u ", m_num_sim_winsn[i] );
   gzprintf(visualizer_file, "\n");
   // warp divergence per shader core
   gzprintf(visualizer_file, "shaderwarpdiv: ");
   for (unsigned i=0;i<m_config->num_shader();i++) 
      gzprintf(visualizer_file, "%u ", m_n_diverge[i] );
   gzprintf(visualizer_file, "\n");
}

void shader_core_stats::print_memory_visualization() const {
    const static bool use_per_kernel = false;
    char buffer[ 255 ];
    buffer[ sizeof( buffer ) - 1 ] = '\0';
    gzFile sharers_visualizer_file;
    gzFile scalar_thread_visualizer_file;
    gzFile warp_touch_visualizer_file;
    gzFile ptx_line_visualizer_file;
    gzFile ptx_line_ex_visualizer_file;
    if ( use_per_kernel ) {
        snprintf( buffer, sizeof( buffer ) - 1, "memory_visualizer_sharers_kernel_%llu.log.gz", g_kernels_launched );
        sharers_visualizer_file = gzopen( buffer, "w" );
        snprintf( buffer, sizeof( buffer ) - 1, "memory_visualizer_scalar_thread_touch_kernel_%llu.log.gz", g_kernels_launched );
        scalar_thread_visualizer_file = gzopen( buffer, "w" );
        snprintf( buffer, sizeof( buffer ) - 1, "memory_visualizer_warp_touch_kernel_%llu.log.gz", g_kernels_launched );
        warp_touch_visualizer_file = gzopen( buffer, "w" );
        snprintf( buffer, sizeof( buffer ) - 1, "memory_visualizer_ptx_line_kernel_%llu.log.gz", g_kernels_launched );
        ptx_line_visualizer_file = gzopen( buffer, "w" );
        snprintf( buffer, sizeof( buffer ) - 1, "memory_visualizer_ptx_line_exclusive_kernel_%llu.log.gz", g_kernels_launched );
        ptx_line_ex_visualizer_file = gzopen( buffer, "w" );
    } else {
        snprintf( buffer, sizeof( buffer ) - 1, "memory_visualizer_sharers.log.gz" );
        sharers_visualizer_file = gzopen( buffer, "w" );
        snprintf( buffer, sizeof( buffer ) - 1, "memory_visualizer_scalar_thread_touch.log.gz" );
        scalar_thread_visualizer_file = gzopen( buffer, "w" );
        snprintf( buffer, sizeof( buffer ) - 1, "memory_visualizer_warp_touch.log.gz" );
        warp_touch_visualizer_file = gzopen( buffer, "w" );
        snprintf( buffer, sizeof( buffer ) - 1, "memory_visualizer_ptx_line.log.gz" );
        ptx_line_visualizer_file = gzopen( buffer, "w" );
        snprintf( buffer, sizeof( buffer ) - 1, "memory_visualizer_ptx_line_exclusive.log.gz" );
        ptx_line_ex_visualizer_file = gzopen( buffer, "w" );
    }

    const new_addr_type smallest_addr = m_block_address_stats[ 0 ].begin()->first;
    unsigned last_one = 0;
    // X axis is the address space
    for ( std::map< new_addr_type, block_access_stats >::const_iterator iter = m_block_address_stats[ 0 ].begin();
            iter != m_block_address_stats[ 0 ].end(); ++iter ) {
        // Have to print out a blank space for portions of the memory space that are unused
        if ( ( iter->first >> 7 ) - ( smallest_addr >> 7 ) != last_one ) {
            unsigned blocks_in_between = ( iter->first >> 7 ) - ( smallest_addr >> 7 ) - last_one;
            // One blank for each block
            for ( unsigned i = 1; i <= blocks_in_between; ++i ) {
                gzprintf(sharers_visualizer_file, "CFLogMem_sharers_s0: ");
                gzprintf(sharers_visualizer_file, "0 0  ");
                gzprintf(scalar_thread_visualizer_file, "CFLogMem_scalar_thread_touch_s0: ");
                gzprintf(scalar_thread_visualizer_file, "0 0  ");
                gzprintf(warp_touch_visualizer_file, "CFLogMem_warp_touch_s0: ");
                gzprintf(warp_touch_visualizer_file, "0 0  ");
                std::list< std::string > ptx_files = get_ptx_filename_list();
                for ( std::list< std::string >::const_iterator iter2 = ptx_files.begin(); iter2 != ptx_files.end(); ++iter2 ) {
                    if ( iter2->size() > 0 ) {
                        gzprintf(ptx_line_visualizer_file, "CFLogMem_ptx_line-%s_hits_s0: ", iter2->c_str());
                        gzprintf(ptx_line_visualizer_file, "0 0  ");
                        gzprintf(ptx_line_ex_visualizer_file, "CFLogMem_ptx_line_exclusive-%s_hits_s0: ", iter2->c_str());
                        gzprintf(ptx_line_ex_visualizer_file, "0 0  ");
                    }
                }
                gzprintf( sharers_visualizer_file, "\nglobalcyclecount: %u\n", last_one + i );
                gzprintf( scalar_thread_visualizer_file, "\nglobalcyclecount: %u\n", last_one + i );
                gzprintf( warp_touch_visualizer_file, "\nglobalcyclecount: %u\n", last_one + i );
                gzprintf( ptx_line_visualizer_file, "\nglobalcyclecount: %u\n", last_one + i );
                gzprintf( ptx_line_ex_visualizer_file, "\nglobalcyclecount: %u\n", last_one + i );
            }
        }
        gzprintf( sharers_visualizer_file, "CFLogMem_sharers_s0: " );
        unsigned total_hits = 1;
        // Sum up all the hits this block has and print it
        for ( std::map< unsigned, unsigned >::const_iterator iter2 = iter->second.warps_hitting_line.begin();
                iter2 != iter->second.warps_hitting_line.end(); ++iter2 ) {
            total_hits += iter2->second;
        }
        gzprintf( sharers_visualizer_file, "%zu %u ", iter->second.warps_hitting_line.size(), total_hits );

        const char* filename = NULL;
        unsigned line_num = 0;
        std::map< std::string, std::string > ptx_file_to_pc_list;
        for ( std::map< unsigned, unsigned >::const_iterator iter2 = iter->second.pcs_hitting_line.begin();
                iter2 != iter->second.pcs_hitting_line.end(); ++iter2 ) {
            get_ptx_source_info( iter2->first, filename, line_num );
            char buff[ 255 ];
            buff[ sizeof( buff ) - 1 ] = '\0';
            snprintf( buff, sizeof( buff ) - 1, "%u %u ", line_num, iter2->second );
            std::string tmp = std::string( buff );
            ptx_file_to_pc_list[ filename ] += tmp;
            total_hits += iter2->second;
        }

        for ( std::map< std::string, std::string >::const_iterator iter2 =  ptx_file_to_pc_list.begin();
                iter2 != ptx_file_to_pc_list.end(); ++iter2 ) {
            gzprintf( ptx_line_visualizer_file, "CFLogMem_ptx_line-%s_hits_s0: %s", iter2->first.c_str(), iter2->second.c_str() );
        }

        ptx_file_to_pc_list.clear();
        for ( std::map< unsigned, unsigned >::const_iterator iter2 = iter->second.pc_exclusive_hits.begin();
                iter2 != iter->second.pc_exclusive_hits.end(); ++iter2 ) {
            get_ptx_source_info( iter2->first, filename, line_num );
            char buff[ 255 ];
            buff[ sizeof( buff ) - 1 ] = '\0';
            snprintf( buff, sizeof( buff ) - 1, "%u %u ", line_num, iter2->second );
            std::string tmp = std::string( buff );
            ptx_file_to_pc_list[ filename ] += tmp;
            total_hits += iter2->second;
        }

        for ( std::map< std::string, std::string >::const_iterator iter2 =  ptx_file_to_pc_list.begin();
                iter2 != ptx_file_to_pc_list.end(); ++iter2 ) {
            gzprintf( ptx_line_ex_visualizer_file, "CFLogMem_ptx_line_exclusive-%s_hits_s0: %s", iter2->first.c_str(), iter2->second.c_str() );
        }

        gzprintf(scalar_thread_visualizer_file, "CFLogMem_scalar_thread_touch_s0: ");
        for ( unsigned i = 0; i < iter->second.threads_hitting_line.size(); ++i ) {
            if ( iter->second.threads_hitting_line[ i ] > 0 ) {
                gzprintf( scalar_thread_visualizer_file, "%u %u ", i, iter->second.threads_hitting_line[ i ] );
            }
        }

        gzprintf(warp_touch_visualizer_file, "CFLogMem_warp_touch_s0: ");
        for( std::map< unsigned, unsigned >::const_iterator iter2 = iter->second.warps_hitting_line.begin();
                iter2 != iter->second.warps_hitting_line.end(); ++iter2 ) {
            gzprintf( warp_touch_visualizer_file, "%u %u ", iter2->first, iter2->second );
        }
        /* TODO - something with these other stats
        std::map< unsigned, std::list< unsigned > > warp_to_hit_times;
        std::map< unsigned, std::list< unsigned > > pc_to_hit_times;*/

        last_one = ( iter->first >> 7 ) - ( smallest_addr >> 7 ) + 1;
        gzprintf( sharers_visualizer_file, "\nglobalcyclecount: %u\n", last_one );
        gzprintf( scalar_thread_visualizer_file, "\nglobalcyclecount: %u\n", last_one );
        gzprintf( warp_touch_visualizer_file, "\nglobalcyclecount: %u\n", last_one );
        gzprintf( ptx_line_visualizer_file, "\nglobalcyclecount: %u\n", last_one );
        gzprintf( ptx_line_ex_visualizer_file, "\nglobalcyclecount: %u\n", last_one );
    }
    gzclose( sharers_visualizer_file );
    gzclose( scalar_thread_visualizer_file );
    gzclose( warp_touch_visualizer_file );
    gzclose( ptx_line_visualizer_file );
    gzclose( ptx_line_ex_visualizer_file );
}

void shader_core_stats::print_recent_warp_issued( FILE *fout, unsigned sid ) const {
	fprintf( fout, "Recently Issued warps for sid %u:\n", sid );
	fprintf( fout, "m_recently_issued_warps=%zu:\n", m_recently_issued_warps[sid].size() );
	for ( std::list< unsigned >::const_iterator it = m_recently_issued_warps[ sid ].begin();
			it != m_recently_issued_warps[ sid ].end(); ++it ) {
		fprintf( fout, "%u, ", *it );
	}
	fprintf( fout, "\n" );
}

#define PROGRAM_MEM_START 0xF0000000 /* should be distinct from other memory spaces... 
                                        check ptx_ir.h to verify this does not overlap 
                                        other memory spaces */
void shader_core_ctx::decode()
{
    if( m_inst_fetch_buffer.m_valid ) {
        // decode 1 or 2 instructions and place them into ibuffer
        address_type pc = m_inst_fetch_buffer.m_pc;
        const warp_inst_t* pI1 = ptx_fetch_inst(pc);
        m_warp[m_inst_fetch_buffer.m_warp_id]->ibuffer_fill(0,pI1);
        m_warp[m_inst_fetch_buffer.m_warp_id]->inc_inst_in_pipeline();
        if( pI1 ) {
           const warp_inst_t* pI2 = ptx_fetch_inst(pc+pI1->isize);
           if( pI2 ) {
               m_warp[m_inst_fetch_buffer.m_warp_id]->ibuffer_fill(1,pI2);
               m_warp[m_inst_fetch_buffer.m_warp_id]->inc_inst_in_pipeline();
           }
        }
        m_inst_fetch_buffer.m_valid = false;
    }
}

std::list< unsigned > shader_core_ctx::get_fetch_order() const {
    std::list< unsigned > ordered_list;
    switch( m_config->gpgpu_instr_fetch_order ) {
        case SHADER_WARP_IFETCH_ORDER_RR: {
            for ( unsigned i = ( m_last_warp_fetched + 1 ) % m_config->max_warps_per_shader, count = 0;
                    count < m_config->max_warps_per_shader;
                    i = ( i + 1 ) % m_config->max_warps_per_shader, ++count ) {
                ordered_list.push_back( i );
            }
        } break;
        case SHADER_WARP_IFETCH_ORDER_GTO: {
            const bool have_any_been_fetched = m_last_warp_fetched > -1;
            if ( have_any_been_fetched ) {
                ordered_list.push_back( m_last_warp_fetched );
            }
            for ( int i = 0; i < static_cast< int >( m_config->max_warps_per_shader ); ++i) {
                if ( i != m_last_warp_fetched || !have_any_been_fetched ) {
                    ordered_list.push_back( i );
                }
            }
        } break;
        default:
            fprintf( stderr, "Unknown instruction fetch ordering gpgpu_instr_fetch_order\n." );
            abort();
    }
    return ordered_list;
}

void shader_core_ctx::fetch()
{
    if( !m_inst_fetch_buffer.m_valid ) {
        // find an active warp with space in instruction buffer that is not already waiting on a cache miss
        // and get next 1-2 instructions from i-cache...
        std::list< unsigned > fetch_list = get_fetch_order();
        assert( fetch_list.size() == m_config->max_warps_per_shader );
        for( std::list< unsigned >::const_iterator it = fetch_list.begin(); it != fetch_list.end(); ++it ) {
            const unsigned warp_id = *it;

            // this code checks if this warp has finished executing and can be reclaimed
            if( m_warp[warp_id]->hardware_done() && !m_scoreboard->pendingWrites(warp_id) && !m_warp[warp_id]->done_exit() ) {
                bool did_exit=false;
                for( unsigned t=0; t<m_config->warp_size;t++) {
                    unsigned tid=warp_id*m_config->warp_size+t;
                    if( m_threadState[tid].m_active == true ) {
                        m_threadState[tid].m_active = false; 
                        unsigned cta_id = m_warp[warp_id]->get_cta_id();
                        register_cta_thread_exit(cta_id);
                        m_not_completed -= 1;
                        m_active_threads.reset(tid);
                        assert( m_thread[tid]!= NULL );
                        did_exit=true;
                    }
                }
                if( did_exit ) {
                    m_warp[warp_id]->set_done_exit();
                    m_scheduling_point_system.inform_warp_exit(warp_id);
                }
            }

            // this code fetches instructions from the i-cache or generates memory requests
            if( !m_warp[warp_id]->functional_done() && !m_warp[warp_id]->imiss_pending() && m_warp[warp_id]->ibuffer_empty() ) {
                address_type pc  = m_warp[warp_id]->get_pc();
                address_type ppc = pc + PROGRAM_MEM_START;
                unsigned nbytes=16; 
                unsigned offset_in_block = pc & (m_config->m_L1I_config.get_line_sz()-1);
                if( (offset_in_block+nbytes) > m_config->m_L1I_config.get_line_sz() )
                    nbytes = (m_config->m_L1I_config.get_line_sz()-offset_in_block);

                // TODO: replace with use of allocator
                // mem_fetch *mf = m_mem_fetch_allocator->alloc()
                mem_access_t acc(INST_ACC_R,ppc,nbytes,false);
                mem_fetch *mf = new mem_fetch(acc,
                                              NULL/*we don't have an instruction yet*/,
                                              READ_PACKET_SIZE,
                                              warp_id,
                                              m_sid,
                                              m_tpc,
                                              m_memory_config );
                std::list<cache_event> events;
                enum cache_request_status status = m_L1I->access( (new_addr_type)ppc, mf, gpu_sim_cycle+gpu_tot_sim_cycle,events);
                if( status == MISS ) {
                    m_last_warp_fetched=warp_id;
                    m_warp[warp_id]->set_imiss_pending();
                    m_warp[warp_id]->set_last_fetch(gpu_sim_cycle);
                } else if( status == HIT ) {
                    m_last_warp_fetched=warp_id;
                    m_inst_fetch_buffer = ifetch_buffer_t(pc,nbytes,warp_id);
                    m_warp[warp_id]->set_last_fetch(gpu_sim_cycle);
                    delete mf;
                } else {
                    assert( status == RESERVATION_FAIL );
                    delete mf;
                }
                break;
            }
        }
    }

    m_L1I->cycle();

    if( m_L1I->access_ready() ) {
        mem_fetch *mf = m_L1I->next_access();
        m_warp[mf->get_wid()]->clear_imiss_pending();
        delete mf;
    }
}

void shader_core_ctx::func_exec_inst( warp_inst_t &inst )
{
    execute_warp_inst_t(inst, m_config->warp_size);
    if( inst.is_load() || inst.is_store() )
        inst.generate_mem_accesses();
}

extern int g_mem_inst_as_nop_perf_mem;
void shader_core_ctx::issue_warp( register_set& pipe_reg_set, const warp_inst_t* next_inst, const active_mask_t &active_mask, unsigned warp_id )
{
    warp_inst_t** pipe_reg = pipe_reg_set.get_free();
    assert(pipe_reg);
    
    m_warp[warp_id]->ibuffer_free();
    m_stats->get_dynamic_per_pc_stats()->log_warp_issue_insn( m_sid, warp_id, next_inst->pc );
    assert(next_inst->valid());
    **pipe_reg = *next_inst; // static instruction information
    (*pipe_reg)->issue( active_mask, warp_id, gpu_tot_sim_cycle + gpu_sim_cycle ); // dynamic instruction information
    m_stats->shader_cycle_distro[2+(*pipe_reg)->active_count()]++;
    func_exec_inst( **pipe_reg );
    if( next_inst->op == BARRIER_OP ) 
        m_barriers.warp_reaches_barrier(m_warp[warp_id]->get_cta_id(),warp_id);
    else if( next_inst->op == MEMORY_BARRIER_OP ) 
        m_warp[warp_id]->set_membar();
    else if( g_mem_inst_as_nop_perf_mem && ( LOAD_OP == next_inst->op || STORE_OP == next_inst->op ) ) {
           const_cast< warp_inst_t * >( next_inst )->op = NO_OP;
    }

    if ( next_inst->op != BARRIER_OP && next_inst->op != MEMORY_BARRIER_OP ) {
        m_scheduling_point_system.signal_inst_issue(warp_id);
    }

    updateSIMTStack(warp_id,m_config->warp_size,*pipe_reg);
    m_scoreboard->reserveRegisters(*pipe_reg);
    m_warp[warp_id]->set_next_pc(next_inst->pc + next_inst->isize);
}

void shader_core_ctx::issue(){
    //really is issue;
    for (unsigned i = 0; i < schedulers.size(); i++) {
        schedulers[i]->cycle();
    }
}

mem_allow_bit_shd_warp_t::mem_allow_bit_shd_warp_t( class shader_core_ctx *shader,
        unsigned warp_size, const cache_conscious_ldst_unit* ldst_unit )
        : base_shd_warp_t( shader, warp_size ), m_ibuffer( IBUFFER_SIZE ), m_ldst_unit( ldst_unit ) {
    for ( unsigned i = 0; i < IBUFFER_SIZE; ++i ) {
        base_shd_warp_t::m_ibuffer[ i ] = m_ibuffer[ i ] = new ibuffer_mem_allow_entry();
    }
}

void mem_allow_bit_shd_warp_t::ibuffer_fill( unsigned slot, const warp_inst_t *pI ) {
   if ( ( ( pI->is_load() || pI->is_store() ) && pI->space.get_type() == global_space )
           && !m_ldst_unit->should_whitelist_warp( m_warp_id ) ) {
       m_ibuffer[slot]->m_mem_allow = false;
   } else {
       m_ibuffer[slot]->m_mem_allow = true;
   }
   base_shd_warp_t::ibuffer_fill( slot, pI );
}

base_shd_warp_t& scheduler_unit::warp(int i){
    return *((*m_warp)[i]);
}

void scheduler_unit::add_supervised_warp_id( int i ) {
    m_supervised_warps.push_back(i);
}

void scheduler_unit::cycle()
{
    bool valid_inst = false;  // there was one warp with a valid instruction to issue (didn't require flush due to control hazard)
    bool ready_inst = false;  // of the valid instructions, there was one not waiting for pending register writes
    bool issued_inst = false; // of these we issued one

    order_warps();
    unsigned issued=0;
    for ( std::vector<int>::const_iterator it = m_supervised_warps.begin(); it != m_supervised_warps.end(); ++it ) {
        unsigned warp_id = *it;
        unsigned checked=0;
        assert( issued == 0 );
        unsigned max_issue = m_shader->get_config()->gpgpu_max_insn_issue_per_warp;
        while( !warp(warp_id).waiting() && !warp(warp_id).ibuffer_empty() && (checked < max_issue) && (checked <= issued) && (issued < max_issue) ) {
            const warp_inst_t *pI = warp(warp_id).ibuffer_next_inst();
            bool valid = warp(warp_id).ibuffer_next_valid();
            bool warp_inst_issued = false;
            unsigned pc,rpc;
            m_simt_stack[warp_id]->get_pdom_stack_top_info(&pc,&rpc);
            if( pI ) {
                assert(valid);
                if( pc != pI->pc ) {
                    // control hazard
                    warp(warp_id).set_next_pc(pc);
                    warp(warp_id).ibuffer_flush();
                } else {
                    valid_inst = true;
                    if ( !m_scoreboard->checkCollision(warp_id, pI) ) {
                        ready_inst = true;
                        const active_mask_t &active_mask = m_simt_stack[warp_id]->get_active_mask();
                        assert( warp(warp_id).inst_in_pipeline() );
                        if ( (pI->op == LOAD_OP) || (pI->op == STORE_OP) || (pI->op == MEMORY_BARRIER_OP) ) {
                            if( m_mem_out->has_free() && !should_stall_mem_op( pI, warp_id )  ) {
                                m_shader->issue_warp(*m_mem_out,pI,active_mask,warp_id);
                                issued++;
                                issued_inst=true;
                                warp_inst_issued = true;
                            }
                        } else {
                            bool sp_pipe_avail = m_sp_out->has_free();
                            bool sfu_pipe_avail = m_sfu_out->has_free();
                            if( sp_pipe_avail && (pI->op != SFU_OP) ) {
                                // always prefer SP pipe for operations that can use both SP and SFU pipelines
                                m_shader->issue_warp(*m_sp_out,pI,active_mask,warp_id);
                                issued++;
                                issued_inst=true;
                                warp_inst_issued = true;
                            } else if ( (pI->op == SFU_OP) || (pI->op == ALU_SFU_OP) ) {
                                if( sfu_pipe_avail ) {
                                    m_shader->issue_warp(*m_sfu_out,pI,active_mask,warp_id);
                                    issued++;
                                    issued_inst=true;
                                    warp_inst_issued = true;
                                }
                            } 
                        }
                    }
                }
            } else if( valid ) {
               // this case can happen after a return instruction in diverged warp
               warp(warp_id).set_next_pc(pc);
               warp(warp_id).ibuffer_flush();
            }
            if(warp_inst_issued)
               warp(warp_id).ibuffer_step();
            checked++;
        }
        if ( issued ) {
        	warp(warp_id).inc_inst_issued( issued );
            m_last_warp_issued=warp_id;
            m_stats->event_warp_issued( m_shader->get_sid(), warp_id, issued, warp(warp_id).get_dynamic_warp_id() );
            break;
        }
    }

    if ( 0 == issued ) {
        m_stats->m_shader_warp_issue_distro[ m_shader->get_sid() ][ 0 ]++;
    }

    m_num_issued_last_cycle = issued;

    post_issue_attempt_process( issued );

    // issue stall statistics:
    if( !valid_inst ) 
        m_stats->shader_cycle_distro[0]++; // idle or control hazard
    else if( !ready_inst ) 
        m_stats->shader_cycle_distro[1]++; // waiting for RAW hazards (possibly due to memory) 
    else if( !issued_inst ) 
        m_stats->shader_cycle_distro[2]++; // pipeline stalled
}

void scheduler_unit::rr_order_warps() {
    m_supervised_warps.clear();
    for ( unsigned supervised_id = (m_last_warp_issued+1) % m_warp->size(), count = 0;
            count < m_warp->size();
            supervised_id = ( supervised_id + 1 ) % m_warp->size(), count++) {
        m_supervised_warps.push_back( supervised_id );
    }
}

void scheduler_unit::greedy_top_order_warps() {
    m_supervised_warps.clear();
    for ( unsigned supervised_id = 0, count = 0;
            count < m_warp->size();
            supervised_id = ( supervised_id + 1 ) % m_warp->size(), count++) {
        m_supervised_warps.push_back( supervised_id );
    }
}

void scheduler_unit::greedy_until_stall_order_warps() {
    m_supervised_warps.clear();
    for ( unsigned supervised_id = m_last_warp_issued, count = 0;
            count < m_warp->size();
            supervised_id = ( supervised_id + 1 ) % m_warp->size(), count++) {
        m_supervised_warps.push_back( supervised_id );
    }
}

bool sort_warps_by_dynamic_id(base_shd_warp_t* lhs, base_shd_warp_t* rhs)
{
    if (rhs && lhs) {
        return lhs->get_dynamic_warp_id() < rhs->get_dynamic_warp_id();
    } else {
        return lhs < rhs;
    }
}

void scheduler_unit::greedy_oldest_order_warps() {
    m_supervised_warps.clear();
    //if ( m_shader->get_sid() == 0 ) printf( "greedy_then_oldest_scheduler_unit::order_warps new warp order\n" );
    m_supervised_warps.push_back( m_last_warp_issued );
    //if ( m_shader->get_sid() == 0 ) printf( "%u, ", m_last_sup_id_issued );
    std::vector<base_shd_warp_t*> oldest_sorted_vector = *m_warp;
    std::sort(oldest_sorted_vector.begin(), oldest_sorted_vector.end(), sort_warps_by_dynamic_id);
    for ( std::vector<base_shd_warp_t*>::const_iterator iter = oldest_sorted_vector.begin();
          iter != oldest_sorted_vector.end(); ++iter ) {
        if ( (*iter)->get_warp_id() != m_last_warp_issued && (*iter)->get_warp_id() < m_shader->get_config()->max_warps_per_shader ) {
            m_supervised_warps.push_back( (*iter)->get_warp_id() );
            //if ( m_shader->get_sid() == 0 ) printf( "%u, ", count );
        }
    }
    //if ( m_shader->get_sid() == 0 ) printf( "\n" );
}

void loose_round_robin_scheduler_unit::order_warps() {
    rr_order_warps();
}

cache_conscious_scheduler_unit::cache_conscious_scheduler_unit(shader_core_stats* stats, shader_core_ctx* shader,
            Scoreboard* scoreboard, simt_stack** simt,
            std::vector<base_shd_warp_t*>* warp,
            register_set* sp_out,
            register_set* sfu_out,
            register_set* mem_out,
            cache_conscious_ldst_unit* dlst_unit,
            const thread_ctx_t* thread,
            unsigned workless_cycles_threshold )
    : scheduler_unit( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out ),
      m_dlst_unit( dlst_unit ), m_thread( thread ), m_cycles_with_no_work_scheduled( 0 ), m_workless_cycles_threshold( workless_cycles_threshold ) {
    dlst_unit->set_scheduler( this );
}

void cache_conscious_scheduler_unit::post_issue_attempt_process( unsigned num_issued ) {
    m_cycles_with_no_work_scheduled = num_issued > 0 ? 0 : m_cycles_with_no_work_scheduled + 1;
    /*
    if ( m_shader->get_sid() == 0 ) {
        printf( "m_cycles_with_no_work_scheduled=%u\n", m_cycles_with_no_work_scheduled );
    }*/
}

void cache_conscious_scheduler_unit::order_warps() {
    static const unsigned num_supervised_warps = m_supervised_warps.size();
    m_supervised_warps.clear();
    for ( unsigned supervised_id = 0, count = 0;
            count < num_supervised_warps;
            supervised_id = ( supervised_id + 1 ) % num_supervised_warps, count++) {

        const active_mask_t &active_mask = m_simt_stack[ supervised_id ]->get_active_mask();
        const warp_inst_t* pI = warp( supervised_id ).ibuffer_next_inst();
        if ( pI && ( pI->is_load() || pI->is_store() ) && pI->space.get_type() == global_space ) {
            bool any_res_fail = false;
            bool all_res_fail = true;
            bool hazard_for_eff_mem = false;
            for ( unsigned i = 0; i < pI->warp_size(); ++i ) {
                if ( active_mask.test( i ) ) {
                    /*
                    if ( m_scoreboard->checkCollision( supervised_id , pI ) ) {
                        hazard_for_eff_mem = true;
                        break;
                    }*/
                    unsigned tid = supervised_id * pI->warp_size() + i;
                    const new_addr_type addr = get_eff_addr( pI->pc, i, m_shader->get_thread_state( tid ) );
                    //printf( "warp( supervised_id ).ibuffer_next_inst()->get_addr( i ) = %p\n", ( void* )addr );
                    //if ( m_dlst_unit->probe_l1( addr ) == RESERVATION_FAIL ) {
                    if ( m_dlst_unit->probe_l1( addr ) == FAIL_PROTECTION ) {
                        any_res_fail = true;
                    } else {
                        all_res_fail = false;
                    }
                }
            }
            //if ( !all_res_fail && !hazard_for_eff_mem ) {
            if ( !any_res_fail && !hazard_for_eff_mem ) {
                m_supervised_warps.push_back( supervised_id );
            } else if ( m_cycles_with_no_work_scheduled > m_workless_cycles_threshold && m_dlst_unit->should_whitelist_warp( supervised_id ) ) {
                m_supervised_warps.push_back( supervised_id );
                //printf( "warp %u was whitelisted\n", supervised_id );
            } else {
                ptx_file_line_add_protective_stall( pI->pc );
            }
        } else {
            m_supervised_warps.push_back( supervised_id );
        }

    }
}

void realistic_cache_conscious_scheduler_unit::print_cc_stats( FILE* file ) const {
    printf( "m_cc_stats.num_times_cache_probed=%u, m_cc_stats.num_times_throttler_accessed=%u, avg_rate=%f\n",
            m_cc_stats.num_times_cache_probed, m_cc_stats.num_times_throttler_accessed,
            ( float ) m_cc_stats.num_times_cache_probed / ( float )m_cc_stats.num_times_throttler_accessed );
}

void realistic_cache_conscious_scheduler_unit::cycle() {
    scheduler_unit::cycle();
    m_num_l1_probes_done_this_cycle = 0;
}

bool realistic_cache_conscious_scheduler_unit::should_stall_mem_op( const warp_inst_t* pI, unsigned warp_id ) {
    if ( pI->space.get_type() == global_space ) {
        if( m_cycles_with_no_work_scheduled > m_workless_cycles_threshold && m_dlst_unit->should_whitelist_warp( warp_id ) ) {
            return false;
        }

        ++m_cc_stats.num_times_throttler_accessed;
        const active_mask_t &active_mask = m_simt_stack[ warp_id ]->get_active_mask();
        for ( unsigned i = 0; i < pI->warp_size() && m_num_l1_probes_done_this_cycle < m_max_cache_probes_per_cycle; ++i ) {
            if ( active_mask.test( i ) ) {
                unsigned tid = warp_id * pI->warp_size() + i;
                const new_addr_type addr = get_eff_addr( pI->pc, i, m_shader->get_thread_state( tid ) );
                ++m_cc_stats.num_times_cache_probed;
                ++m_num_l1_probes_done_this_cycle;
                if ( m_dlst_unit->probe_l1( addr ) == FAIL_PROTECTION ) {
                    return true;
                }
            }
        }
    }
    if ( m_num_l1_probes_done_this_cycle == m_max_cache_probes_per_cycle ) {
        //printf( "Probed L1 Too Many Times\n" );
    }
    return false;
}

void realistic_cache_conscious_scheduler_unit::order_warps() {
    if ( GREEDY_THEN_OLDEST_WARP_SCHEDULER_POLICY == m_warp_scheduling_policy ) {
        greedy_oldest_order_warps();
    } else if( GREEDY_TOP_WARP_SCHEDULER_POLICY == m_warp_scheduling_policy ) {
        greedy_top_order_warps();
    } else if( ROUND_ROBIN_WARP_SCHEDULER_POLICY == m_warp_scheduling_policy ) {
        rr_order_warps();
    } else {
        fprintf( stderr, "Error, Unknown Scheduling Policy\n" );
    }
}

void realistic_memory_allow_bit_cc_scheduler_unit::cycle() {
    for ( unsigned warp_id = ( m_last_warp_processed + 1 ) % m_warp->size(), count = 0;
            count < m_warp->size(); warp_id = ( warp_id + 1 ) % m_warp->size(), ++count ) {
        bool done = false;
        const warp_inst_t *pI = warp( warp_id ).ibuffer_next_inst();
        unsigned pc,rpc;
        m_simt_stack[warp_id]->get_pdom_stack_top_info(&pc,&rpc);
        if ( pI && pc == pI->pc && !m_scoreboard->checkCollision( warp_id, pI )
                && !dynamic_cast< mem_allow_bit_shd_warp_t& >( warp( warp_id ) ).check_mem_allow() ) {
            const active_mask_t &active_mask = m_simt_stack[ warp_id ]->get_active_mask();
            unsigned num_lanes_okay = 0;
            for ( unsigned lane = 0; lane < m_shader->get_config()->warp_size
                && m_num_l1_probes_done_this_cycle < m_max_cache_probes_per_cycle; ++lane ) {
                if ( active_mask.test( lane ) ) {
                    unsigned tid = warp_id * m_shader->get_config()->warp_size + lane;
                    const new_addr_type addr = get_eff_addr( pI->pc,
                            lane, m_shader->get_thread_state( tid ) );
                    ++m_cc_stats.num_times_cache_probed;
                    ++m_num_l1_probes_done_this_cycle;
                    if ( m_dlst_unit->probe_l1( addr ) != FAIL_PROTECTION ) {
                        ++num_lanes_okay;
                    }
                }
            }

            if ( num_lanes_okay == m_num_l1_probes_done_this_cycle || active_mask.count() == 0 ) {
                dynamic_cast< mem_allow_bit_shd_warp_t& >( warp( warp_id ) ).set_mem_allow();
            }

            m_last_warp_processed = warp_id;
            done = true;
            break;
        }
        if ( done ) {
            break;
        }
    }

    // Check the SO
    if ( is_stall_override_in_effect() ) {
        for ( unsigned warp_id = 0; warp_id < m_warp->size(); ++warp_id ) {
            if( m_dlst_unit->should_whitelist_warp( warp_id ) ) {
                for ( unsigned i = 0; i < 2; ++i ) {
                    dynamic_cast< mem_allow_bit_shd_warp_t& >( warp( warp_id ) ).set_mem_allow();
                    warp( warp_id ).ibuffer_step();
                }
            }
        }
    }

    ++m_cc_stats.num_times_throttler_accessed;
    realistic_cache_conscious_scheduler_unit::cycle();
}

bool realistic_memory_allow_bit_cc_scheduler_unit::should_stall_mem_op( const warp_inst_t* pI, unsigned warp_id ) {
    return !dynamic_cast< mem_allow_bit_shd_warp_t& >( warp( warp_id ) ).check_mem_allow();
}

void greepy_top_scheduler_unit::order_warps() {
    greedy_top_order_warps();

}

// The point of this is to ensure we maximize parallelism such that we keep the working set below the l1 cache size
void custom_memcached_scheduler_unit::order_warps() {
    // Okay, this is a little hacky, but making this static since the size of the supervised warps changes during execution, but all I want is the original value - TODO set this properly
    static const unsigned num_supervised_warps = m_supervised_warps.size();
    unsigned num_warps_in_crit_loop = 0;
    unsigned cache_needed = 0;
    std::list< unsigned > warps_in_crit_loop;
    for ( unsigned i = 0; i < num_supervised_warps; ++i ) {
        // Count up the number of warps that are in the critical memcached loop
        if ( warp( i ).get_pc() >= 5632 && warp( i ).get_pc() <= 5784 ) {
            cache_needed += AMMOUNT_ADDED_BY_THREAD * warp( i ).get_active_threads();
            ++num_warps_in_crit_loop;
            warps_in_crit_loop.push_back( i );
        }
    }

    m_supervised_warps.clear();
    unsigned warps_to_schedule = num_supervised_warps;
    if ( cache_needed > m_l1_cache_size ) {
        //printf( "shader %d hit the maximum with %d warps in the critical loop, cache needed = %u\n", m_shader->get_sid(), num_warps_in_crit_loop, cache_needed );
        unsigned cache_handed_out = 0;
        for ( std::list< unsigned >::const_iterator it = warps_in_crit_loop.begin(); 
                it != warps_in_crit_loop.end() && cache_handed_out < m_l1_cache_size; 
                ++it ) {
            cache_handed_out += AMMOUNT_ADDED_BY_THREAD * warp( *it ).get_active_threads();
            m_supervised_warps.push_back( *it );
        }
    } else {
        for ( unsigned count = 0; count < warps_to_schedule; count++ ) {
            m_supervised_warps.push_back( count );
        }
    }
}

void greedy_until_stall_scheduler_unit::order_warps() {
    greedy_until_stall_order_warps();
}

void nvidia_isca_2011_two_level_warp_scheduler::order_warps() {
    // This scheduler maintains a list of active warps which remain active until an instruction dependent on a
    // long latency operation is encountered (source operand is in global memory or is texture based)
    m_supervised_warps.clear();
    for ( std::list< unsigned >::const_iterator it = m_active_warps_list.begin();
            it != m_active_warps_list.end(); ++it ) {
        m_supervised_warps.push_back( *it );
    }
}

void nvidia_isca_2011_two_level_warp_scheduler::post_issue_attempt_process( unsigned num_issued ) {
    //if ( num_issued > 0 ) {
        for ( std::list< unsigned >::iterator it = m_active_warps_list.begin();
                    it != m_active_warps_list.end(); ++it ) {
            if ( ( !warp( *it ).functional_done()
                    && ( nvidia_isca_2011_is_sourced_by_long_lat( warp( *it ).get_pc() )
                            || m_shader->warp_waiting_at_barrier( *it )
                            || m_shader->warp_waiting_at_mem_barrier( *it ) ) )
                    || warp( *it ).done_exit() ) {
                m_pending_warps_list.push_back( *it );
                m_last_warp_issued = *it = m_pending_warps_list.front();
                m_pending_warps_list.pop_front();
                break;
            }
        }

        // Now reorder your active warps based on the scheduling policy
        std::list< unsigned > new_active_warps;
        if ( ROUND_ROBIN_WARP_SCHEDULER_POLICY == m_policy_type ) {
            for ( std::list< unsigned >::reverse_iterator rit = m_active_warps_list.rbegin();
                            rit != m_active_warps_list.rend(); ++rit ) {
                if ( *rit == static_cast< unsigned >( m_last_warp_issued ) ) {
                    break;
                } else {
                    new_active_warps.push_front( *rit );
                }
            }
            for ( std::list< unsigned >::iterator it = m_active_warps_list.begin();
                            it != m_active_warps_list.end(); ++it ) {
                if ( *it == static_cast< unsigned >( m_last_warp_issued ) ) {
                    break;
                } else {
                    new_active_warps.push_back( *it );
                }
            }
            new_active_warps.push_back( m_last_warp_issued );
            // Based on what they have in their paper, greedy is actually greedy then round robin
        } else if ( GREEDY_THEN_ROUND_ROBIN_SCHEDULER_POLICY == m_policy_type ) {
            for ( std::list< unsigned >::reverse_iterator rit = m_active_warps_list.rbegin();
                            rit != m_active_warps_list.rend(); ++rit ) {
                if ( *rit == static_cast< unsigned >( m_last_warp_issued ) ) {
                    break;
                } else {
                    new_active_warps.push_front( *rit );
                }
            }
            for ( std::list< unsigned >::iterator it = m_active_warps_list.begin();
                            it != m_active_warps_list.end(); ++it ) {
                if ( *it == static_cast< unsigned >( m_last_warp_issued ) ) {
                    break;
                } else {
                    new_active_warps.push_back( *it );
                }
            }
            new_active_warps.push_front( m_last_warp_issued );
        } else {
            fprintf( stderr, "Error - nvidia_isca_2011_two_level_warp_scheduler::post_issue_attempt_process unimplemented policy\n" );
            abort();
        }

        m_active_warps_list = new_active_warps;
        assert( m_active_warps_list.size() == m_active_warps );
        //if ( m_shader->get_sid() == 0 )print_active_warp_list( stdout );
        //if ( m_shader->get_sid() == 0 )print_pending_warp_list( stdout );
        //if ( m_shader->get_sid() == 0 )print_scheduled_list( stdout );
    //}
}

void greedy_then_oldest_scheduler_unit::order_warps() {
    greedy_oldest_order_warps();
}


concurrency_limited_scheduler_unit::concurrency_limited_scheduler_unit( shader_core_stats* stats, shader_core_ctx* shader,
        Scoreboard* scoreboard, simt_stack** simt,
        std::vector<base_shd_warp_t*>* warp,
        register_set* sp_out,
        register_set* sfu_out,
        register_set* mem_out,
        unsigned num_concurrent_warps,
        unsigned num_concurrent_ctas,
        unsigned warp_size )
    : scheduler_unit( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out ), m_max_concurrent_warps( num_concurrent_warps ), m_max_concurrent_ctas( num_concurrent_ctas ), m_warp_size( warp_size ) {
    // One of these has to mutually exclusively non-zero
    assert(  ( m_max_concurrent_ctas > 0 ) ^ ( m_max_concurrent_warps > 0 ) );
}

void concurrency_limited_scheduler_unit::order_warps() {
    if ( m_max_concurrent_warps > 0 ) {
        m_supervised_warps.clear();
        std::vector<base_shd_warp_t*> oldest_sorted_vector = *m_warp;
        std::sort(oldest_sorted_vector.begin(), oldest_sorted_vector.end(), sort_warps_by_dynamic_id);
        for ( std::vector<base_shd_warp_t*>::const_iterator iter = oldest_sorted_vector.begin();
            iter != oldest_sorted_vector.end(); ++iter ) {
            if ( (*iter)->get_warp_id() < m_shader->get_config()->max_warps_per_shader ) {
                if ( !(*iter)->done_exit() ) {
                    m_supervised_warps.push_back( (*iter)->get_warp_id() );
                }
                if ( m_supervised_warps.size() == m_max_concurrent_warps
                    && !m_shader->warp_waiting_at_barrier( (*iter)->get_warp_id() )
                    && !m_shader->warp_waiting_at_mem_barrier( (*iter)->get_warp_id() ) ) {
                    break;
                }
            }
        }
       //assert( m_supervised_warps.size() <= m_max_concurrent_warps );
    } else {
        assert( m_max_concurrent_ctas > 0 );
        std::set< unsigned > running_ctas;
        m_supervised_warps.clear();
        for ( unsigned count = 0, warp_id = m_last_warp_issued; count < m_warp_size;
                ++count, warp_id = ( warp_id + 1 ) % m_warp_size ) {
            if ( !warp( count ).done_exit()
                    && ( running_ctas.size() < m_max_concurrent_ctas
                            || running_ctas.find( warp( count ).get_cta_id() ) != running_ctas.end() ) ) {
                running_ctas.insert( warp( count ).get_cta_id() );
                m_supervised_warps.push_back( count );
            }
        }
        assert( running_ctas.size() <= m_max_concurrent_ctas );
    }
}

texas_tech_report_scheduler_unit::texas_tech_report_scheduler_unit( shader_core_stats* stats, shader_core_ctx* shader,
                Scoreboard* scoreboard, simt_stack** simt,
                std::vector<base_shd_warp_t*>* warp,
                register_set* sp_out,
                register_set* sfu_out,
                register_set* mem_out,
                unsigned fetch_group_size ) : scheduler_unit( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out ), m_fetch_group_size( fetch_group_size ) {
    assert ( m_warp->size() % fetch_group_size == 0 );
}

void texas_tech_report_scheduler_unit::order_warps() {
    m_supervised_warps.clear();

    const unsigned last_fetch_group_scheduled = m_last_warp_issued / m_fetch_group_size;
    const unsigned total_num_fetch_groups = m_warp->size() / m_fetch_group_size;
    for ( unsigned current_fetch_group = last_fetch_group_scheduled % m_fetch_group_size, group_count = 0;
            group_count < total_num_fetch_groups;
            current_fetch_group = ( current_fetch_group + 1 ) % total_num_fetch_groups, ++group_count ) {
        for ( unsigned fg_id = ( m_last_warp_issued + 1 ) % m_fetch_group_size, count = 0;
                count < m_fetch_group_size;
                ++count, fg_id = ( fg_id + 1 ) % m_fetch_group_size ) {
            const unsigned warp_id = current_fetch_group * m_fetch_group_size + fg_id;
            assert( warp_id < m_warp->size() );
            m_supervised_warps.push_back( warp_id );
        }
    }
}

// This scheduler insures that each warp will issue in order even if it means
void strict_round_robin_scheduler_unit::order_warps() {
    m_supervised_warps.clear();

    if ( m_num_issued_last_cycle > 0 || warp( m_current_turn_warp ).done_exit() || warp( m_current_turn_warp ).waiting() ) {
        for ( unsigned warp_id = ( m_current_turn_warp + 1 ) % m_warp->size(), count = 0;
                count < m_warp->size();
                warp_id = ( warp_id + 1 ) % m_warp->size(), ++count ) {
            if ( !warp( warp_id ).done_exit() && !warp( warp_id ).waiting() ) {
                m_supervised_warps.push_back( warp_id );
                m_current_turn_warp = warp_id;
                break;
            }
        }
    } else {
        m_supervised_warps.push_back( m_current_turn_warp );
    }
}

void point_system_scheduler_unit::order_warps()
{
    if ( m_point_system->get_state() == SCHED_STATE_NORMAL ) {
        greedy_oldest_order_warps();
    } else if ( m_point_system->get_state() == SCHED_STATE_LOAD_EXCLUSIVE_LIST ) {
        greedy_oldest_order_warps();
        // If the warp is not in our exclusive load list AND it is trying to issue a load that would pollute the cache, pluck it out
        std::vector<int> new_list;
        std::vector<int>::iterator curr_warp = m_supervised_warps.begin();
        while ( curr_warp != m_supervised_warps.end() ) {
            const bool is_warp_blacklisted = ( warp(*curr_warp).ibuffer_next_inst() && warp(*curr_warp).ibuffer_next_inst()->is_load() 
                && ( warp(*curr_warp).ibuffer_next_inst()->space.get_type () == local_space || warp(*curr_warp).ibuffer_next_inst()->space.get_type() == global_space ) 
                && !m_point_system->is_in_exclusive_list(*curr_warp));
            if (!is_warp_blacklisted) {
                new_list.push_back(*curr_warp);
            }
            ++curr_warp;
        }
        m_supervised_warps = new_list;
    }
}

void shader_core_ctx::read_operands()
{
}

address_type coalesced_segment(address_type addr, unsigned segment_size_lg2bytes)
{
   return  (addr >> segment_size_lg2bytes);
}

// Returns numbers of addresses in translated_addrs, each addr points to a 4B (32-bit) word
unsigned shader_core_ctx::translate_local_memaddr( address_type localaddr, unsigned tid, unsigned num_shader, unsigned datasize, new_addr_type* translated_addrs )
{
   // During functional execution, each thread sees its own memory space for local memory, but these
   // need to be mapped to a shared address space for timing simulation.  We do that mapping here.

   address_type thread_base = 0;
   unsigned max_concurrent_threads=0;
   if (m_config->gpgpu_local_mem_map) {
      // Dnew = D*N + T%nTpC + nTpC*C
      // N = nTpC*nCpS*nS (max concurent threads)
      // C = nS*K + S (hw cta number per gpu)
      // K = T/nTpC   (hw cta number per core)
      // D = data index
      // T = thread
      // nTpC = number of threads per CTA
      // nCpS = number of CTA per shader
      // 
      // for a given local memory address threads in a CTA map to contiguous addresses,
      // then distribute across memory space by CTAs from successive shader cores first, 
      // then by successive CTA in same shader core
      thread_base = 4*(kernel_padded_threads_per_cta * (m_sid + num_shader * (tid / kernel_padded_threads_per_cta))
                       + tid % kernel_padded_threads_per_cta); 
      max_concurrent_threads = kernel_padded_threads_per_cta * kernel_max_cta_per_shader * num_shader;
   } else {
      // legacy mapping that maps the same address in the local memory space of all threads 
      // to a single contiguous address region 
      thread_base = 4*(m_config->n_thread_per_shader * m_sid + tid);
      max_concurrent_threads = num_shader * m_config->n_thread_per_shader;
   }
   assert( thread_base < 4/*word size*/*max_concurrent_threads );

   // If requested datasize > 4B, split into multiple 4B accesses
   // otherwise do one sub-4 byte memory access
   unsigned num_accesses = 0;

   if(datasize >= 4) {
      // >4B access, split into 4B chunks
      assert(datasize%4 == 0);   // Must be a multiple of 4B
      num_accesses = datasize/4;
      assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD); // max 32B
      assert(localaddr%4 == 0); // Address must be 4B aligned - required if accessing 4B per request, otherwise access will overflow into next thread's space
      for(unsigned i=0; i<num_accesses; i++) {
          address_type local_word = localaddr/4 + i;
          address_type linear_address = local_word*max_concurrent_threads*4 + thread_base + LOCAL_GENERIC_START;
          translated_addrs[i] = linear_address;
      }
   } else {
      // Sub-4B access, do only one access
      assert(datasize > 0);
      num_accesses = 1;
      address_type local_word = localaddr/4;
      address_type local_word_offset = localaddr%4;
      assert( (localaddr+datasize-1)/4  == local_word ); // Make sure access doesn't overflow into next 4B chunk
      address_type linear_address = local_word*max_concurrent_threads*4 + local_word_offset + thread_base + LOCAL_GENERIC_START;
      translated_addrs[0] = linear_address;
   }
   return num_accesses;
}

/////////////////////////////////////////////////////////////////////////////////////////
int shader_core_ctx::test_res_bus(int latency){
	for(int i=0; i<num_result_bus; i++){
		if(!m_result_bus[i]->test(latency)){return i;}
	}
	return -1;
}

void shader_core_ctx::execute()
{
	for(int i=0; i<num_result_bus; i++){
		*(m_result_bus[i]) >>=1;
	}
    for( unsigned n=0; n < m_num_function_units; n++ ) {
        unsigned multiplier = m_fu[n]->clock_multiplier();
        for( unsigned c=0; c < multiplier; c++ ) 
            m_fu[n]->cycle();
        enum pipeline_stage_name_t issue_port = m_issue_port[n];
        register_set& issue_inst = m_pipeline_reg[ issue_port ];
	warp_inst_t** ready_reg = issue_inst.get_ready();
        if( issue_inst.has_ready() && m_fu[n]->can_issue( **ready_reg ) ) {
            bool schedule_wb_now = !m_fu[n]->stallable();
            int resbus = -1;
            if( schedule_wb_now && (resbus=test_res_bus( (*ready_reg)->latency ))!=-1 ) {
                assert( (*ready_reg)->latency < MAX_ALU_LATENCY );
                m_result_bus[resbus]->set( (*ready_reg)->latency );
                m_fu[n]->issue( issue_inst );
            } else if( !schedule_wb_now ) {
                m_fu[n]->issue( issue_inst );
            } else {
                // stall issue (cannot reserve result bus)
            }
        }
    }
    m_scheduling_point_system.cycle();
}

void ldst_unit::collect_mem_divergence_statistics( const warp_inst_t &inst, const mem_access_t &access )
{
	//for now only measure useful global reads
	if (access.get_type()==GLOBAL_ACC_R or access.get_type()==LOCAL_ACC_R) {
	  m_stats->old_tot_useful_bytes_read += access.get_byte_mask().count();
	  m_stats->old_tot_bytes_read += access.get_size();
	}
}

mem_div_status ldst_unit::mem_div_read_byte(const new_addr_type address, const bool using_now )
{
    m_stats->tot_bytes_read++;
    bool new_or_too_old = false;
    //std::map <new_addr_type/*byte address*/,div_map_t/*used before?*/ >::iterator i=m_stats->mem_div_map.find(address);
    std::map <new_addr_type, div_map_t>::iterator i = mem_div_map_ldst.find(address);
    div_map_t *entry = NULL;

    if ( i == mem_div_map_ldst.end() ) {
       new_or_too_old = true;
       entry = &mem_div_map_ldst[address];

       //entry->cycle_loaded = (gpu_tot_sim_cycle + gpu_sim_cycle);
	entry->cycle_loaded = m_stats->m_div_info.back().cycle_start;
       entry->status = MEM_DIV_NEVERUSED;
       entry->num_access = 1;
       m_stats->total_unique_bytes_read++;
    } else {
        entry = &i->second;
	new_or_too_old = false;
        //if( (gpu_sim_cycle+gpu_tot_sim_cycle) - entry->last_access_cycle >= m_config->mem_div_cycles_threshold )
        // new_or_too_old = true;
     }
    if ( new_or_too_old ) {
       if (using_now) {
           m_stats->tot_useful_bytes_read++;
	   m_stats->bytes_used_now++;
           entry->status = MEM_DIV_USEDATARRIVAL;
       } else {
           entry->status = MEM_DIV_NEVERUSED;
	   m_stats->m_div_info.back().never_used_bytes++;
       }
    } else if (using_now) { // it is already in the map and being used now
        if(entry->status == MEM_DIV_NEVERUSED) { // and not used before
            m_stats->tot_eventually_useful_bytes_read++;
            entry->status = MEM_DIV_USEDLATER;//record it as used   
          
	    for(unsigned i=0; i<m_stats->m_div_info.size(); i++){
		//assert(entry->cycle_loaded != m_stats->m_div_info.at(i).cycle_start && entry->cycle_loaded != (gpu_tot_sim_cycle + gpu_sim_cycle));
//		assert(entry->cycle_loaded != (gpu_tot_sim_cycle + gpu_sim_cycle));
               	//if(entry->cycle_loaded >= m_stats->m_div_info.at(i).cycle_start && (entry->cycle_loaded < (m_stats->m_div_info.at(i).cycle_end == 0 ? 
		//							(gpu_tot_sim_cycle + gpu_sim_cycle) : m_stats->m_div_info.at(i).cycle_end))){
		if(entry->cycle_loaded == m_stats->m_div_info.at(i).cycle_start){
			m_stats->m_div_info.at(i).useful_later_bytes++;
			assert(m_stats->m_div_info.at(i).never_used_bytes != 0);
			m_stats->m_div_info.at(i).never_used_bytes--;
			break;
		}
            }
	    m_stats->m_div_info.back().useful_later_bytes_read_now++;
        } else { //hiting on something that was used before 
            if ( entry->status == MEM_DIV_USEDATARRIVAL ) {
                m_stats->tot_already_useful_bytes_read++;
		m_stats->already_bytes_used_now++;

		entry->num_access++;
            } else {
                m_stats->tot_already_eventually_useful_bytes_read++;
  		m_stats->m_div_info.back().useful_later_bytes_read_now++;
                for(unsigned i=0; i<m_stats->m_div_info.size(); i++){
                	if(entry->cycle_loaded >= m_stats->m_div_info.at(i).cycle_start && (entry->cycle_loaded < (m_stats->m_div_info.at(i).cycle_end == 0 ? 
									(gpu_tot_sim_cycle + gpu_sim_cycle) : m_stats->m_div_info.at(i).cycle_end))){
                		m_stats->m_div_info.at(i).already_useful_later_bytes++;
				break;
			}
                }
            }
        }
    } else {
        // Not using something that is already in the cache
        // its status should NOT change!
        // but how about its access time ? the last access gets updated for now!
    }
    assert(entry->status!=MEM_DIV_INVALID);
    return entry->status;
}

void ldst_unit::mem_div_new_access( const mem_access_t &access, addr_t access_pc, unsigned long warp_id )
{
   //for now only measure useful global reads
   if( access.get_type()==GLOBAL_ACC_R or access.get_type()==LOCAL_ACC_R ) {
      const bool is_divergent = access.is_diverged();
      m_stats->mem_div_accesses++;

      if( is_divergent )
         m_stats->mem_div_divergent_accesses++;

      m_stats->mem_div_access_num_bytes_requested += access.get_size();

      // classification stats
      mem_div_load_counters& load_counters = m_stats->warp_to_pc_to_load_counters_map[ warp_id ][ access_pc ];
      load_counters.num_accesses++;
      m_stats->all_warps_pc_to_load_counters_map[ access_pc ].num_accesses++;
      unsigned bytes_with_some_use = 0;
      const mem_access_byte_mask_t &mask = access.get_byte_mask();

      if(access.get_type() == GLOBAL_ACC_R){
      	    m_stats->t_bytes_read += 64;
     	    m_stats->t_mem_access++;
	    int index = ((float)mask.count()*8.0)/64.0;
	    if(index == 8)
		index = 7;
      	    m_stats->t_bytes_used[index] += mask.count();
      }


      if(!m_stats->m_div_info.size() || m_stats->m_div_info.back().num_mem_access >= 500){
       	  // Add new bin to the mem divergence info
	  if(m_stats->m_div_info.size())
		  m_stats->m_div_info.back().cycle_end = (gpu_tot_sim_cycle + gpu_sim_cycle);

       	  mem_div_info temp;
	  temp.cycle_start = (gpu_tot_sim_cycle + gpu_sim_cycle);
	  temp.cycle_end = 0;
       	  temp.total_bytes = 0;
	  temp.total_unique_bytes = 0;
	  temp.useful_later_bytes_read_now = 0;
	  for(int i=0; i<8; i++){
	       	  temp.useful_bytes[i] = 0;
		  temp.already_useful_bytes[i] = 0;
	  }
  	  temp.num_mem_access = 0;
          temp.useful_later_bytes = 0;
	  temp.already_useful_later_bytes = 0;
	  temp.never_used_bytes = 0;
          m_stats->m_div_info.push_back(temp);
      }
      m_stats->m_div_info.back().num_mem_access++;

      unsigned start_byte = 0;
      m_stats->total_bytes_read = access.get_size();
      m_stats->total_unique_bytes_read = 0;
      m_stats->bytes_used_now = 0;
      m_stats->already_bytes_used_now = 0;

      if(access.is_byte_mask_flag())
	      start_byte = access.get_byte_mask_start();

      for( unsigned b=start_byte; b < start_byte+m_stats->total_bytes_read; b++ ) {
          address_type addr = access.get_addr() + (b-start_byte);
	  assert((b-start_byte) >= 0);
          mem_div_status status = mem_div_read_byte( (new_addr_type)addr, mask.test(b) );

          if ( status == MEM_DIV_USEDATARRIVAL || MEM_DIV_USEDLATER == status )
        	  bytes_with_some_use++;
      }
 
      m_stats->m_div_info.back().total_bytes += m_stats->total_bytes_read;
      m_stats->m_div_info.back().total_unique_bytes += m_stats->total_unique_bytes_read;


      int index = 0;
      index = (( (float)m_stats->bytes_used_now*8.0)/ (float)m_stats->total_bytes_read);// - 1;
      //index = (( (float)m_stats->bytes_used_now*8.0)/ 128.0);
      if(index == 8)
	index--;
      m_stats->m_div_info.back().useful_bytes[ index ] += m_stats->bytes_used_now;

      index = (( (float)m_stats->already_bytes_used_now*8.0)/ (float)m_stats->total_bytes_read);// - 1;
      //index = (( (float)m_stats->already_bytes_used_now*8.0)/128.0);
      if(index == 8)
      	index--;
      m_stats->m_div_info.back().already_useful_bytes[ index ] += m_stats->already_bytes_used_now;


      if ( bytes_with_some_use == mask.size() ) {
    	  load_counters.num_times_all_bytes_used_or_eventually_used++;
    	  m_stats->all_warps_pc_to_load_counters_map[ access_pc ].num_times_all_bytes_used_or_eventually_used++;
      }
      else if ( bytes_with_some_use >= mask.size() * 3 / 4 ) {
    	  load_counters.num_times_gte_3_quarters_bytes_used_or_eventually_used++;
    	  m_stats->all_warps_pc_to_load_counters_map[ access_pc ].num_times_gte_3_quarters_bytes_used_or_eventually_used++;
      }
      else if ( bytes_with_some_use >= mask.size() / 2 ) {
         load_counters.num_times_gte_half_bytes_used_or_eventually_used++;
         m_stats->all_warps_pc_to_load_counters_map[ access_pc ].num_times_gte_half_bytes_used_or_eventually_used++;
      }
      else if ( bytes_with_some_use >= mask.size() / 4 ) {
    	  load_counters.num_times_gte_1_quarter_used_or_eventually_used++;
    	  m_stats->all_warps_pc_to_load_counters_map[ access_pc ].num_times_gte_1_quarter_used_or_eventually_used++;
      }
   }
}

void ldst_unit::print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses ) {
   if ( m_L1D_per_warp ) {
       for ( unsigned i = 0; i < m_config->max_warps_per_shader; ++i ) {
           m_L1D_per_warp[ i ]->print( fp, dl1_accesses, dl1_misses );
       }
   } else if ( m_L1D ) {
       m_L1D->print( fp, dl1_accesses, dl1_misses );
   }
   m_tlb->print( fp );
   if ( m_vm_manager )
       m_vm_manager->print( fp );
}

void shader_core_ctx::warp_inst_complete(const warp_inst_t &inst)
{
   #if 0
      printf("[warp_inst_complete] uid=%u core=%u warp=%u pc=%#x @ time=%llu issued@%llu\n", 
             inst.get_uid(), m_sid, inst.warp_id(), inst.pc, gpu_tot_sim_cycle + gpu_sim_cycle, inst.get_issue_cycle()); 
   #endif
   m_stats->m_num_sim_insn[m_sid] += inst.active_count();
   m_stats->m_num_sim_winsn[m_sid]++;
   m_gpu->gpu_sim_insn += inst.active_count();
   inst.completed(gpu_tot_sim_cycle + gpu_sim_cycle); 
}

void shader_core_ctx::writeback()
{
    warp_inst_t** preg = m_pipeline_reg[EX_WB].get_ready();
    warp_inst_t* pipe_reg = (preg==NULL)? NULL:*preg;
    while( preg and !pipe_reg->empty() ) {
    	/*
    	 * Right now, the writeback stage drains all waiting instructions
    	 * assuming there are enough ports in the register file or the
    	 * conflicts are resolved at issue.
    	 */
    	/*
    	 * The operand collector writeback can generally generate a stall
    	 * However, here, the pipelines should be un-stallable. This is
    	 * guaranteed because this is the first time the writeback function
    	 * is called after the operand collector's step function, which
    	 * resets the allocations. There is one case which could result in
    	 * the writeback function returning false (stall), which is when
    	 * an instruction tries to modify two registers (GPR and predicate)
    	 * To handle this case, we ignore the return value (thus allowing
    	 * no stalling).
    	 */
        m_operand_collector.writeback(*pipe_reg);
        unsigned warp_id = pipe_reg->warp_id();
        m_scoreboard->releaseRegisters( pipe_reg );
        m_warp[warp_id]->dec_inst_in_pipeline();
        warp_inst_complete(*pipe_reg); 
        m_gpu->gpu_sim_insn_last_update_sid = m_sid;
        m_gpu->gpu_sim_insn_last_update = gpu_sim_cycle;
        m_last_inst_gpu_sim_cycle = gpu_sim_cycle;
        m_last_inst_gpu_tot_sim_cycle = gpu_tot_sim_cycle;
        ptx_file_line_stats_add_latency( pipe_reg->pc, gpu_tot_sim_cycle + gpu_sim_cycle - pipe_reg->get_issue_cycle() );
        pipe_reg->clear();
        preg = m_pipeline_reg[EX_WB].get_ready();
        pipe_reg = (preg==NULL)? NULL:*preg;
    }
}

bool ldst_unit::shared_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type)
{
   if( inst.space.get_type() != shared_space )
       return true;
   bool stall = inst.dispatch_delay();
   if( stall ) {
       fail_type = S_MEM;
       rc_fail = BK_CONF;
   } else 
       rc_fail = NO_RC_FAIL;
   return !stall; 
}

void ldst_unit::use_shared_cache_system( data_cache* dcache, warp_inst_t &inst, cache_request_status& status, mem_fetch *mf, std::list<cache_event> &events ) {
    unsigned dummy;
    new_addr_type evicted_block_addr = 0xDEADBEEF;
    // If we are not going to hit in the normal cache, then try the shared cache
    cache_request_status dcache_probe_result = dcache->probe( mf->get_addr(), evicted_block_addr );
    if ( dcache_probe_result != HIT ) {
        if ( m_shared_data_only_L1D->probe( mf->get_addr(), dummy ) == HIT ) {
            status = m_shared_data_only_L1D->access( mf->get_addr(), gpu_sim_cycle + gpu_tot_sim_cycle, dummy, mf );
            assert( HIT == status );
            m_stats->m_num_shared_only_cache_hits[ m_sid ]++;
        } else {
            status = dcache->access(mf->get_addr(),mf,gpu_sim_cycle+gpu_tot_sim_cycle,events);
        }
        //  If the evicted block is shared then push it into the special shared cache
        if ( dcache_probe_result == MISS ) {
            assert( evicted_block_addr != 0xDEADBEEF );
            if ( m_stats->m_block_address_stats[ m_sid ].find( evicted_block_addr ) != m_stats->m_block_address_stats[ m_sid ].end()
                && m_stats->m_block_address_stats[ m_sid ][ evicted_block_addr ].warps_hitting_line.size() > m_config->gpgpu_shared_only_sharers_for_shared ) {
                if ( m_shared_data_only_L1D->probe( evicted_block_addr, dummy ) != HIT ) {
                    m_shared_data_only_L1D->fill( evicted_block_addr, gpu_sim_cycle + gpu_tot_sim_cycle );
                    m_stats->m_num_shared_only_cache_insertions[ m_sid ]++;
                } else {
                    m_shared_data_only_L1D->access( evicted_block_addr, gpu_sim_cycle + gpu_tot_sim_cycle, dummy, mf );
                }
            }
        }
    } else {
        status = dcache->access(mf->get_addr(),mf,gpu_sim_cycle+gpu_tot_sim_cycle,events);
    }
}

mem_stage_stall_type ldst_unit::process_memory_access_queue( cache_t *cache, warp_inst_t &inst )
{
    mem_stage_stall_type result = NO_RC_FAIL;
    if( inst.accessq_empty() )
        return result;

    const mem_access_t &access = inst.accessq_back();
    collect_mem_divergence_statistics( inst, access );

    vm_page_mapping_payload* mapping_for_access = NULL;
    if ( DYNAMIC_TRANSLATION == g_translation_config && inst.accessq_back().get_vm_page() ) {
       mapping_for_access = new vm_page_mapping_payload( *inst.accessq_back().get_vm_page() );
    }
    mem_fetch *mf = m_mf_allocator->alloc(inst, inst.accessq_back(), mapping_for_access );
	
	mf->set_sector_mask(inst.accessq_back().get_sector()); // Set the accessed sectors for sectored cache

    std::list<cache_event> events;
    extern int g_perfect_pointer_cache;

    data_cache* dcache = dynamic_cast< data_cache* >( cache );
    enum cache_request_status status;
    if ( g_perfect_pointer_cache && access.is_ptr() ) {
        status = HIT;
    } else if ( dynamic_cast< high_locality_protected_cache* >( cache ) ) {
        status = dynamic_cast< high_locality_protected_cache* >( cache )->access(mf->get_addr(),mf,gpu_sim_cycle+gpu_tot_sim_cycle,events, inst);
    } else if ( !m_config->m_L1D_shared_only_config.disabled() && dcache ) {
        use_shared_cache_system( dcache, inst, status, mf, events );
    } else {
        status = cache->access(mf->get_addr(),mf,gpu_sim_cycle+gpu_tot_sim_cycle,events);
    }

    if ( inst.space.get_type() == global_space || inst.space.get_type() == local_space ) {
        ptx_file_line_add_l1_access( inst.pc );
        if ( RESERVATION_FAIL == status || MISS == status ) {
            ptx_file_line_add_l1_miss( inst.pc );
        } else if ( HIT == status ) {
            m_stats->record_cache_block_hit( m_config->m_L1D_config.block_addr( mf->get_addr() ),
                    m_sid,
                    inst.warp_id(),
                    inst.pc,
                    access.get_warp_mask() );
        }

        if ( was_vc_hit( events ) ) {
            ++m_stats->m_shader_vc_hit_distro[ m_sid ][ inst.warp_id() ];
        }

        ++m_stats->m_shader_l1_data_cache_distro[ m_sid ][ status ];
        if (0==m_sid) {
            ++m_stats->m_shader_dl1_s0_per_warp_distro[ inst.warp_id() ][ status ];
        }

        if ( m_sid == 0 ) {
            m_stats->get_dynamic_per_pc_stats()->log_warp_l1_data_cache_access( inst.warp_id(), inst.pc );
            if ( status == HIT ) {
                m_stats->get_dynamic_per_pc_stats()->log_warp_l1_data_cache_hit( inst.warp_id(), inst.pc );
            }
        }
    }

    bool write_sent = was_write_sent(events);
    bool read_sent = was_read_sent(events);

    if( write_sent ) 
        m_core->inc_store_req( inst.warp_id() );
    if ( status == HIT ) {
        assert( !read_sent );
#if ENABLE_MEM_DIV_STATS
        mem_div_new_access( access, inst.pc, inst.warp_id() );
#endif
        inst.accessq_pop_back();
        if (inst.is_load()) {
            m_hit_queue.push_back( std::pair<mem_fetch*,int>( mf, L1_HIT_LATENCY ) );
        }
    } else if ( status == RESERVATION_FAIL ) {
        result = COAL_STALL;
        assert( !read_sent );
        assert( !write_sent );
        delete mf;
    } else {
        assert( status == MISS || status == HIT_RESERVED );
#if ENABLE_MEM_DIV_STATS
        mem_div_new_access( access, inst.pc, inst.warp_id() );
#endif
        //inst.clear_active( access.get_warp_mask() ); // threads in mf writeback when mf returns 
        inst.accessq_pop_back();
    }

    if( !inst.accessq_empty() )
        result = BK_CONF;
    return result;
}

bool ldst_unit::constant_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type)
{
   if( inst.empty() || ((inst.space.get_type() != const_space) && (inst.space.get_type() != param_space_kernel)) )
       return true;
   if( inst.active_count() == 0 ) 
       return true;
   mem_stage_stall_type fail = process_memory_access_queue(m_L1C,inst);
   if (fail != NO_RC_FAIL){ 
      rc_fail = fail; //keep other fails if this didn't fail.
      fail_type = C_MEM;
      if (rc_fail == BK_CONF or rc_fail == COAL_STALL) {
         m_stats->gpgpu_n_cmem_portconflict++; //coal stalls aren't really a bank conflict, but this maintains previous behavior.
      }
   }
   return inst.accessq_empty(); //done if empty.
}

bool ldst_unit::texture_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type)
{
   if( inst.empty() || inst.space.get_type() != tex_space )
       return true;
   if( inst.active_count() == 0 ) 
       return true;
   mem_stage_stall_type fail = process_memory_access_queue(m_L1T,inst);
   if (fail != NO_RC_FAIL){ 
      rc_fail = fail; //keep other fails if this didn't fail.
      fail_type = T_MEM;
   }
   return inst.accessq_empty(); //done if empty.
}

bool ldst_unit::memory_cycle( warp_inst_t &inst, mem_stage_stall_type &stall_reason, mem_stage_access_type &access_type )
{
   if( inst.empty() || 
       ((inst.space.get_type() != global_space) &&
        (inst.space.get_type() != local_space) &&
        (inst.space.get_type() != param_space_local)) ) 
       return true;
   if( inst.active_count() == 0 ) 
       return true;
   assert( !inst.accessq_empty() );

   mem_stage_stall_type stall_cond = NO_RC_FAIL;
   const mem_access_t &access = inst.accessq_back();
   unsigned size = access.get_size(); 

   if( CACHE_GLOBAL == inst.cache_op || (m_L1D == NULL) ) {
       // bypass L1 cache
       if( m_icnt->full(size, inst.is_store() || inst.isatomic()) ) {
           stall_cond = ICNT_RC_FAIL;
       } else {
           mem_fetch *mf = m_mf_allocator->alloc(inst,access);
           m_icnt->push(mf);

           inst.accessq_pop_back();
           //inst.clear_active( access.get_warp_mask() );

           if( inst.is_load() ) { 
              for( unsigned r=0; r < 4; r++) 
                  if(inst.out[r] > 0) 
                      assert( m_pending_writes[inst.warp_id()][inst.out[r]] > 0 );
           } else if( inst.is_store() ) 
              m_core->inc_store_req( inst.warp_id() );
       }
   } else {
       assert( CACHE_UNDEFINED != inst.cache_op );
       if ( m_L1D_per_warp ) {
           stall_cond = process_memory_access_queue( m_L1D_per_warp[ inst.warp_id() ], inst );
       } else {
           stall_cond = process_memory_access_queue(m_L1D,inst);
       }
   }
   if( !inst.accessq_empty() ) 
       stall_cond = COAL_STALL;
   if (stall_cond != NO_RC_FAIL) {
      stall_reason = stall_cond;
      bool iswrite = inst.is_store();
      if (inst.space.is_local()) 
         access_type = (iswrite)?L_MEM_ST:L_MEM_LD;
      else 
         access_type = (iswrite)?G_MEM_ST:G_MEM_LD;
   }
   return inst.accessq_empty(); 
}


bool ldst_unit::response_buffer_full() const
{
    return m_response_fifo.size() >= m_config->ldst_unit_response_queue_size;
}

void ldst_unit::fill( mem_fetch *mf )
{
    mf->set_status(IN_SHADER_LDST_RESPONSE_FIFO,gpu_sim_cycle+gpu_tot_sim_cycle);
    m_response_fifo.push_back(mf);
}

void ldst_unit::flush()
{
    if ( m_L1D  ) {
        m_L1D->flush();
    }

    if ( m_L1C ) {
        m_L1C->flush();
    }

    if ( m_L1D_per_warp ) {
        for ( unsigned i = 0; i < m_config->max_warps_per_shader; ++i ) {
            m_L1D_per_warp[ i ]->flush();
        }
    }

    if ( m_tlb ) {
        m_tlb->flush();
    }
}

simd_function_unit::simd_function_unit( const shader_core_config *config )
{ 
    m_new_dispatch_reg = false;
    m_config=config;
    m_dispatch_reg = new warp_inst_t(config); 
}

sfu::sfu( register_set* result_port, const shader_core_config *config ) 
    : pipelined_simd_unit(result_port,config,config->max_sfu_latency) 
{ 
    m_name = "SFU"; 
}

sp_unit::sp_unit( register_set* result_port, const shader_core_config *config ) 
    : pipelined_simd_unit(result_port,config,config->max_sp_latency) 
{ 
    m_name = "SP "; 
}


pipelined_simd_unit::pipelined_simd_unit( register_set* result_port, const shader_core_config *config, unsigned max_latency ) 
    : simd_function_unit(config) 
{
    m_result_port = result_port;
    m_pipeline_depth = max_latency;
    m_pipeline_reg = new warp_inst_t*[m_pipeline_depth];
    for( unsigned i=0; i < m_pipeline_depth; i++ ) 
	m_pipeline_reg[i] = new warp_inst_t( config );
}

ldst_unit::ldst_unit( mem_fetch_interface *icnt,
                      shader_core_mem_fetch_allocator *mf_allocator,
                      shader_core_ctx *core, 
                      opndcoll_rfu_t *operand_collector,
                      Scoreboard *scoreboard,
                      const shader_core_config *config,
                      const memory_config *mem_config,  
                      shader_core_stats *stats, 
                      unsigned sid,
                      unsigned tpc ) : pipelined_simd_unit(NULL,config,3), m_next_wb(config), m_vm_manager( NULL )
{
    m_memory_config = mem_config;
    m_icnt = icnt;
    m_mf_allocator=mf_allocator;
    m_core = core;
    m_operand_collector = operand_collector;
    m_scoreboard = scoreboard;
    m_stats = stats;
    m_sid = sid;
    m_tpc = tpc;
    #define STRSIZE 1024
    char L1T_name[STRSIZE];
    char L1C_name[STRSIZE];
    char L1D_name[STRSIZE];
    char TLB_name[STRSIZE];
    snprintf(L1T_name, STRSIZE, "L1T_%03d", m_sid);
    snprintf(L1C_name, STRSIZE, "L1C_%03d", m_sid);
    snprintf(L1D_name, STRSIZE, "L1D_%03d", m_sid);
    snprintf(TLB_name, STRSIZE, "TLB_%03d", m_sid);
    m_L1T = new tex_cache(L1T_name,m_config->m_L1T_config,m_sid,get_shader_texture_cache_id(),icnt,IN_L1T_MISS_QUEUE,IN_SHADER_L1T_ROB);
    m_L1C = new read_only_cache(L1C_name,m_config->m_L1C_config,m_sid,get_shader_constant_cache_id(),icnt,IN_L1C_MISS_QUEUE, false);

    m_L1D = NULL;
    m_tlb = NULL;

    if( !m_config->m_L1D_config.disabled() ) {
        switch ( m_config->m_L1D_config.get_cache_type() ) {
        case CACHE_NORMAL:
            m_L1D = new data_cache(L1D_name,m_config->m_L1D_config,m_sid,get_shader_normal_cache_id(),m_icnt,m_mf_allocator,IN_L1D_MISS_QUEUE, m_config->m_L1D_config.is_access_dump_enabled());
            break;
        case CACHE_SECTORED:
            m_L1D = new sector_cache(L1D_name,m_config->m_L1D_config,m_sid,get_shader_normal_cache_id(),m_icnt,m_mf_allocator,IN_L1D_MISS_QUEUE, false);
            break;
        case CACHE_VIRTUAL_POLICY_MANAGED:
            m_L1D = new virtual_policy_cache(L1D_name,m_config->m_L1D_config,m_sid,get_shader_normal_cache_id(),m_icnt,m_mf_allocator,IN_L1D_MISS_QUEUE, false);
            break;
        case CACHE_HIGH_LOCALITY_PROTECTED:
            m_L1D = new high_locality_protected_cache(L1D_name,m_config->m_L1D_config,m_sid,get_shader_normal_cache_id(),m_icnt,m_mf_allocator,IN_L1D_MISS_QUEUE, m_config->m_high_locality_L1D_config, m_core->get_config()->max_warps_per_shader, m_config->m_L1DVC_config, false );
            break;
        default:
            printf("Unknown L1 Config\n");
            abort();
        }
    }

    if ( !m_config->m_L1D_shared_only_config.disabled() ) {
        m_shared_data_only_L1D = new tag_array(m_config->m_L1D_shared_only_config,m_sid,get_shader_normal_cache_id(),false);
    }

    switch( m_config->m_TLB_config.get_translation_config() )
    {
    case NO_TRANSLATION:
        m_tlb = new shader_ideal_tlb( TLB_name, m_config->m_TLB_config, m_sid, get_shader_tlb_id(), icnt, IN_TLB_MISS_QUEUE, m_core->get_gpu() );
        break;
    case STATIC_TRANSLATION:
        m_tlb = new shader_page_reordering_tlb( TLB_name, m_config->m_TLB_config, m_sid, get_shader_tlb_id(), icnt, IN_TLB_MISS_QUEUE, m_core->get_gpu(), NULL, NULL );
        break;
    case DYNAMIC_TRANSLATION: {
        virtual_policy_cache* vm_policy_cache = dynamic_cast< virtual_policy_cache* >( m_L1D );
        assert( vm_policy_cache ); // Using dynamic translation requires that the L1 cache be virtual policy managed
        m_vm_manager = new vm_policy_manager( m_config->m_vm_manager_config,
                        m_mf_allocator,
                        icnt,
                        IN_VM_MANAGER_QUEUE,
                        sid,
                        m_L1D->get_line_sz(),
                        &m_core->get_gpu()->get_page_factory() );
        m_tlb = new shader_page_reordering_tlb( TLB_name, m_config->m_TLB_config, m_sid, get_shader_tlb_id(), icnt, IN_TLB_MISS_QUEUE, m_core->get_gpu(), m_vm_manager, vm_policy_cache );
    } break;
    default:
       // Unkown TLB config
       assert(0);
    }
    m_mem_rc = NO_RC_FAIL;
    m_num_writeback_clients=5; // = shared memory, global/local (uncached), L1D, L1T, L1C
    m_writeback_arb = 0;
    m_next_global=NULL;
    m_last_inst_gpu_sim_cycle=0;
    m_last_inst_gpu_tot_sim_cycle=0;

    // This is a mythical configuration designed to find out how much intra-thread locality is present in applications
    m_L1D_per_warp = NULL;
    if ( !m_config->m_L1D_per_warp_individual_cache.disabled() ) {
        m_L1D_per_warp = static_cast< evict_on_write_cache ** >( calloc( m_config->max_warps_per_shader, sizeof( evict_on_write_cache * ) ) );
        for ( unsigned i = 0; i < m_config->max_warps_per_shader; ++i ) {
            snprintf( L1D_name, STRSIZE, "L1D_per_warp_%03d_thread_%03d", m_sid, i );
            m_L1D_per_warp[ i ] = new data_cache(L1D_name,m_config->m_L1D_per_warp_individual_cache,m_sid,get_shader_normal_cache_id(),m_icnt,m_mf_allocator,IN_L1D_MISS_QUEUE, false);
        }
    }
}

void ldst_unit::writeback()
{
    // process next instruction that is going to writeback
    if( !m_next_wb.empty() ) {
        if( m_operand_collector->writeback(m_next_wb) ) {
            bool insn_completed = false; 
            for( unsigned r=0; r < 4; r++ ) {
                if( m_next_wb.out[r] > 0 ) {
                    if( m_next_wb.space.get_type() != shared_space ) {
                        assert( m_pending_writes[m_next_wb.warp_id()][m_next_wb.out[r]] > 0 );
                        unsigned still_pending = --m_pending_writes[m_next_wb.warp_id()][m_next_wb.out[r]];
                        if( !still_pending ) {
                            m_pending_writes[m_next_wb.warp_id()].erase(m_next_wb.out[r]);
                            m_scoreboard->releaseRegister( m_next_wb.warp_id(), m_next_wb.out[r] );
                            insn_completed = true; 
                        }
                    } else { // shared 
                        m_scoreboard->releaseRegister( m_next_wb.warp_id(), m_next_wb.out[r] );
                        insn_completed = true; 
                    }
                }
            }
            if( insn_completed ) {
                m_core->warp_inst_complete(m_next_wb); 
            }
            m_next_wb.clear();
            m_last_inst_gpu_sim_cycle = gpu_sim_cycle;
            m_last_inst_gpu_tot_sim_cycle = gpu_tot_sim_cycle;
            ptx_file_line_stats_add_latency( m_next_wb.pc, gpu_tot_sim_cycle + gpu_sim_cycle - m_next_wb.get_issue_cycle() );
        }
    }
    
    std::list< std::pair<mem_fetch*,int> >::iterator it = m_hit_queue.begin();
    while ( it != m_hit_queue.end() ) {
        --(it->second);
        if ( it->second == 0 ) {
            for( unsigned r=0; r < 4; r++ ) {
                const int wid = it->first->get_wid();
                const int reg_num = it->first->get_inst().out[r];
                if ( reg_num > 0 ) {
                    unsigned still_pending = --m_pending_writes[wid][reg_num];
                    if( !still_pending ) {
                        m_pending_writes[wid].erase(reg_num);
                        m_scoreboard->releaseRegister( wid, reg_num );
                        m_stats->m_num_sim_insn[m_sid]++;
                        m_core->get_gpu()->gpu_sim_insn += m_next_wb.active_count();
                    }
                }
            }
            delete it->first;
            it = m_hit_queue.erase(it);
        } else {
            ++it;
        }
    }

    unsigned serviced_client = -1; 
    for( unsigned c = 0; m_next_wb.empty() && (c < m_num_writeback_clients); c++ ) {
        unsigned next_client = (c+m_writeback_arb)%m_num_writeback_clients;
        switch( next_client ) {
        case 0: // shared memory 
            if( !m_pipeline_reg[0]->empty() ) {
                m_next_wb = *m_pipeline_reg[0];
                m_core->dec_inst_in_pipeline(m_pipeline_reg[0]->warp_id());
                m_pipeline_reg[0]->clear();
                serviced_client = next_client; 
            }
            break;
        case 1: // texture response
            if( m_L1T->access_ready() ) {
                mem_fetch *mf = m_L1T->next_access();
                m_next_wb = mf->get_inst();
                delete mf;
                serviced_client = next_client; 
            }
            break;
        case 2: // const cache response
            if( m_L1C->access_ready() ) {
                mem_fetch *mf = m_L1C->next_access();
                m_next_wb = mf->get_inst();
                delete mf;
                serviced_client = next_client; 
            }
            break;
        case 3: // global/local
            if( m_next_global ) {
                m_next_wb = m_next_global->get_inst();
                if( m_next_global->isatomic() ) 
                    m_core->decrement_atomic_count(m_next_global->get_wid(),m_next_global->get_access_warp_mask().count());
                delete m_next_global;
                m_next_global = NULL;
                serviced_client = next_client; 
            }
            break;
        case 4: 
            if ( m_L1D_per_warp ) {
                for ( unsigned i = 0; i < m_config->max_warps_per_shader; ++i ) {
                    if ( m_L1D_per_warp[ i ]->access_ready() ) {
                        mem_fetch *mf = m_L1D_per_warp[ i ]->next_access();
                        m_next_wb = mf->get_inst();
                        delete mf;
                        break;
                    }
                }
            } else if( m_L1D && m_L1D->access_ready() ) {
                mem_fetch *mf = m_L1D->next_access();
                m_next_wb = mf->get_inst();
                delete mf;
                serviced_client = next_client; 
            }
            break;
        default: abort();
        }
    }
    // update arbitration priority only if: 
    // 1. the writeback buffer was available 
    // 2. a client was serviced 
    if (serviced_client != (unsigned)-1) {
        m_writeback_arb = (serviced_client + 1) % m_num_writeback_clients; 
    }
}

unsigned ldst_unit::clock_multiplier() const
{ 
    return m_config->mem_warp_parts; 
}

void ldst_unit::issue( register_set &reg_set )
{
	warp_inst_t* inst = *(reg_set.get_ready());
   // stat collection
   m_core->mem_instruction_stats(*inst); 

   // record how many pending register writes/memory accesses there are for this instruction 
   assert(inst->empty() == false); 
   if (inst->is_load() and inst->space.get_type() != shared_space) {
      unsigned warp_id = inst->warp_id(); 
      unsigned n_accesses = inst->accessq_count(); 
      for (unsigned r = 0; r < 4; r++) {
         unsigned reg_id = inst->out[r]; 
         if (reg_id > 0) {
            m_pending_writes[warp_id][reg_id] += n_accesses; 
         }
      }
   }

   pipelined_simd_unit::issue(reg_set);
}



void print_cache(data_cache *d){
	// Prints Info about data_cache d. 
	d->print_data_cache_info();
}

bool simplified_mem_system_ldst_unit::memory_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &access_type) {
    if( inst.empty() ||
        ((inst.space.get_type() != global_space) &&
        (inst.space.get_type() != local_space) &&
        (inst.space.get_type() != param_space_local)) ) {
            return true;
    }
    if( inst.active_count() == 0 ) {
        return true;
    }
    assert( !inst.accessq_empty() );

    const mem_access_t &access = inst.accessq_back();

    mem_fetch* mf = m_mf_allocator->alloc( inst, access );
    if ( m_simplified_mem_system->add_memory_request( m_sid, mf ) ) {
        if( inst.is_load() ) {
            for( unsigned r=0; r < 4; r++) {
                if( inst.out[r] > 0 ) {
                    m_pending_writes[inst.warp_id()][inst.out[r]]++;
                }
            }
        } else if( inst.is_store() ) {
            m_core->inc_store_req( inst.warp_id() );
        }
        inst.accessq_pop_back();
        if ( !inst.accessq_empty() ) {
            rc_fail = COAL_STALL;
        }
    } else {
        delete mf;
        rc_fail = ICNT_RC_FAIL; // Since the memory system is really simple, we will just call our our stalls a general BW issue
    }
    const bool iswrite = inst.is_store();
    if ( inst.space.is_local() ) {
        access_type = ( iswrite ) ? L_MEM_ST : L_MEM_LD;
    }
    else {
        access_type = ( iswrite ) ? G_MEM_ST : G_MEM_LD;
    }
    return inst.accessq_empty();
}

void simplified_mem_system_ldst_unit::cycle() {
    m_simplified_mem_system->get_my_completed_requests( m_sid, m_response_fifo );
    ldst_unit::cycle();
}

void ldst_unit::cycle()
{
   writeback();
   m_operand_collector->step();
   for( unsigned stage=0; (stage+1)<m_pipeline_depth; stage++ ) 
       if( m_pipeline_reg[stage]->empty() && !m_pipeline_reg[stage+1]->empty() )
            move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage+1]);

    if( !m_response_fifo.empty() ) {
        mem_fetch *mf = m_response_fifo.front();
        if (mf->istexture()) {
            m_L1T->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);
            m_response_fifo.pop_front();
        } else if (mf->isconst())  {
            mf->set_status(IN_SHADER_FETCHED,gpu_sim_cycle+gpu_tot_sim_cycle);
            m_L1C->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);
            m_response_fifo.pop_front();
        } else {
            if( mf->get_type() == PROTOCOL_MSG ) {
                if( mf->get_access_type() == VM_POLICY_CHANGE_ACK_RC )
                    m_vm_manager->fill(mf);
                else if( mf->get_access_type() == VM_POLICY_CHANGE_REQ_RC )
                    m_vm_manager->fill(mf);
                else abort();
                m_response_fifo.pop_front();
                delete mf;
            } else if( mf->get_type() == WRITE_ACK ||
                    ( m_config->gpgpu_perfect_mem && mf->get_is_write() ) ) {
                m_core->store_ack(mf);
                m_response_fifo.pop_front();
                delete mf;
            } else {
                assert( !mf->get_is_write() ); // L1 cache is write evict, allocate line on load miss only
                if ( m_config->m_simplified_config.enabled ) {
                    if ( m_next_global == NULL ) {
                        mf->set_status(IN_SHADER_FETCHED,gpu_sim_cycle+gpu_tot_sim_cycle);
                        m_response_fifo.pop_front();
                        m_next_global = mf;
                    }
                } else if( mf->get_inst().cache_op != CACHE_GLOBAL && m_L1D ) {
                    if ( m_L1D_per_warp ) {
                        m_L1D_per_warp[ mf->get_wid() ]->fill( mf,gpu_sim_cycle+gpu_tot_sim_cycle );
                    } else {
                        m_L1D->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);
                    }
                    m_response_fifo.pop_front();
                } else if( m_next_global == NULL ) {
                    mf->set_status(IN_SHADER_FETCHED,gpu_sim_cycle+gpu_tot_sim_cycle);
                    m_response_fifo.pop_front();
                    m_next_global = mf;
                }
            }
        }
    }

    m_tlb->cycle();
    if ( DYNAMIC_TRANSLATION == g_translation_config ) {
        vm_page_mapping_payload *change = m_vm_manager->cycle();
        if( change )
            dynamic_cast<shader_page_reordering_tlb*>(m_tlb)->insert_new_policy( *change );
    }

    warp_inst_t &pipe_reg = *m_dispatch_reg;
    if ( !pipe_reg.empty() && m_new_dispatch_reg && ( pipe_reg.is_load() || pipe_reg.is_store() )  ) {
        if ( DYNAMIC_TRANSLATION == g_translation_config ) {
            m_vm_manager->access( pipe_reg, gpu_sim_cycle+gpu_tot_sim_cycle, *dynamic_cast< shader_page_reordering_tlb* >( m_tlb ) ); // send access to VM policy manager for this shader core
        }

        // translate your address to linear space.
        // the current setup assumes a physically indexed L1 cache for simplicity
        const cache_request_status tlb_result = m_tlb->translate_addrs( pipe_reg );
        assert( HIT == tlb_result ); // Right now we don't deal with anything other than a tlb hit

        // tgrogers - this was moved here for the VM policy mappign stuff so we could decide if we wanted to remap things before generating the
        // memory accesses.  Integrating Fermi CL 11956 requires that this be done at functional execution.
        // Therefore; if we ever want the VM mapping stuff to work again, we need to fix it so we can delay the coalescing...
        //pipe_reg.generate_mem_accesses();
    }

    m_L1T->cycle();
    m_L1C->cycle();
    if ( m_L1D_per_warp ) {
        // This is going to increase our Injection BW into the interconnect, but for now I just care about miss rates with this per-warp cache.
        for ( unsigned i = 0; i < m_config->max_warps_per_shader; ++i ) {
            m_L1D_per_warp[ i ]->cycle();
        }
    } else if( m_L1D ) {
        m_L1D->cycle();
    }

    enum mem_stage_stall_type rc_fail = NO_RC_FAIL;
    mem_stage_access_type type;
    bool done = true;
    done &= shared_cycle(pipe_reg, rc_fail, type);
    done &= constant_cycle(pipe_reg, rc_fail, type);
    done &= texture_cycle(pipe_reg, rc_fail, type);
    done &= memory_cycle(pipe_reg, rc_fail, type);
    m_mem_rc = rc_fail;

    if (!done) { // log stall types and do nothing else
        assert(rc_fail != NO_RC_FAIL);
        m_stats->gpgpu_n_stall_shd_mem++;
        m_stats->gpu_stall_shd_mem_breakdown[type][rc_fail]++;
    } else if (!pipe_reg.empty()) {
        unsigned warp_id = pipe_reg.warp_id();
        if (pipe_reg.is_load()) {
            if (pipe_reg.space.get_type() == shared_space) {
                if (m_pipeline_reg[2]->empty()) {
                    // new shared memory request
                    move_warp(m_pipeline_reg[2], m_dispatch_reg);
                    m_dispatch_reg->clear();
                }
            } else {
                //if( pipe_reg.active_count() > 0 ) {
                //    if( !m_operand_collector->writeback(pipe_reg) )
                //        return;
                //}

               bool pending_requests=false;
               for( unsigned r=0; r<4; r++ ) {
                   unsigned reg_id = pipe_reg.out[r];
                   if( reg_id > 0 ) {
                       if( m_pending_writes[warp_id].find(reg_id) != m_pending_writes[warp_id].end() ) {
                           if ( m_pending_writes[warp_id][reg_id] > 0 ) {
                               pending_requests=true;
                               break;
                           } else {
                               // this instruction is done already
                               m_pending_writes[warp_id].erase(reg_id); 
                           }
                       }
                   }
               }
               if( !pending_requests ) {
                   m_core->warp_inst_complete(*m_dispatch_reg); 
                   m_scoreboard->releaseRegisters(m_dispatch_reg);
               }
               m_core->dec_inst_in_pipeline(warp_id);
               m_dispatch_reg->clear();
           }
       } else {
           // stores exit pipeline here
           m_core->dec_inst_in_pipeline(warp_id);
           m_core->warp_inst_complete(*m_dispatch_reg); 
           m_dispatch_reg->clear();
       }
   }
   m_new_dispatch_reg = false;
}

void shader_core_ctx::register_cta_thread_exit( unsigned cta_num )
{
   assert( m_cta_status[cta_num] > 0 );
   m_cta_status[cta_num]--;
   if (!m_cta_status[cta_num]) {
      m_n_active_cta--;
      m_barriers.deallocate_barrier(cta_num);
      shader_CTA_count_unlog(m_sid, 1);
      printf("GPGPU-Sim uArch: Shader %d finished CTA #%d (%lld,%lld), %u CTAs running\n", m_sid, cta_num, gpu_sim_cycle, gpu_tot_sim_cycle,
             m_n_active_cta );
      if( m_n_active_cta == 0 ) {
          assert( m_kernel != NULL );
          m_kernel->dec_running();
          printf("GPGPU-Sim uArch: Shader %u empty (release kernel %u \'%s\').\n", m_sid, m_kernel->get_uid(),
                 m_kernel->name().c_str() );
          if( m_kernel->no_more_ctas_to_run() ) {
              if( !m_kernel->running() ) {
                  printf("GPGPU-Sim uArch: GPU detected kernel \'%s\' finished on shader %u.\n", m_kernel->name().c_str(), m_sid );
                  m_gpu->set_kernel_done( m_kernel );
              }
          }
          m_kernel=NULL;
          fflush(stdout);
      }
   }
}

void gpgpu_sim::shader_print_runtime_stat( FILE *fout ) 
{
    /*
   fprintf(fout, "SHD_INSN: ");
   for (unsigned i=0;i<m_n_shader;i++) 
      fprintf(fout, "%u ",m_sc[i]->get_num_sim_insn());
   fprintf(fout, "\n");
   fprintf(fout, "SHD_THDS: ");
   for (unsigned i=0;i<m_n_shader;i++) 
      fprintf(fout, "%u ",m_sc[i]->get_not_completed());
   fprintf(fout, "\n");
   fprintf(fout, "SHD_DIVG: ");
   for (unsigned i=0;i<m_n_shader;i++) 
      fprintf(fout, "%u ",m_sc[i]->get_n_diverge());
   fprintf(fout, "\n");

   fprintf(fout, "THD_INSN: ");
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_insn(i) );
   fprintf(fout, "\n");
   */
}


void gpgpu_sim::shader_print_l1_miss_stat( FILE *fout ) const
{
   unsigned total_d1_misses = 0, total_d1_accesses = 0;
   for ( unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i ) {
         unsigned custer_d1_misses = 0, cluster_d1_accesses = 0;
         m_cluster[ i ]->print_cache_stats( fout, cluster_d1_accesses, custer_d1_misses );
         total_d1_misses += custer_d1_misses;
         total_d1_accesses += cluster_d1_accesses;
   }
   fprintf( fout, "total_dl1_misses=%d\n", total_d1_misses );
   fprintf( fout, "total_dl1_accesses=%d\n", total_d1_accesses );
   fprintf( fout, "total_dl1_miss_rate= %f\n", (float)total_d1_misses / (float)total_d1_accesses );
   fprintf( fout, "total_dl1_mpki= %f\n", (float)total_d1_misses / (float) gpu_tot_sim_insn * 1000.0f );
   fprintf( fout, "total_dl1_mpki_ptx= %f\n", (float)total_d1_misses / (float) g_ptx_sim_num_insn * 1000.0f );
   /*
   fprintf(fout, "THD_INSN_AC: ");
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_insn_ac(i));
   fprintf(fout, "\n");
   fprintf(fout, "T_L1_Mss: "); //l1 miss rate per thread
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_mis_ac(i));
   fprintf(fout, "\n");
   fprintf(fout, "T_L1_Mgs: "); //l1 merged miss rate per thread
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_mis_ac(i) - m_sc[0]->get_thread_n_l1_mrghit_ac(i));
   fprintf(fout, "\n");
   fprintf(fout, "T_L1_Acc: "); //l1 access per thread
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) 
      fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_access_ac(i));
   fprintf(fout, "\n");

   //per warp
   int temp =0; 
   fprintf(fout, "W_L1_Mss: "); //l1 miss rate per warp
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
      temp += m_sc[0]->get_thread_n_l1_mis_ac(i);
      if (i%m_shader_config->warp_size == (unsigned)(m_shader_config->warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");
   temp=0;
   fprintf(fout, "W_L1_Mgs: "); //l1 merged miss rate per warp
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
      temp += (m_sc[0]->get_thread_n_l1_mis_ac(i) - m_sc[0]->get_thread_n_l1_mrghit_ac(i) );
      if (i%m_shader_config->warp_size == (unsigned)(m_shader_config->warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");
   temp =0;
   fprintf(fout, "W_L1_Acc: "); //l1 access per warp
   for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
      temp += m_sc[0]->get_thread_n_l1_access_ac(i);
      if (i%m_shader_config->warp_size == (unsigned)(m_shader_config->warp_size-1)) {
         fprintf(fout, "%d ", temp);
         temp = 0;
      }
   }
   fprintf(fout, "\n");
   */
}

void warp_inst_t::print( FILE *fout ) const
{
    if (empty() ) {
        fprintf(fout,"bubble\n" );
        return;
    } else 
        fprintf(fout,"0x%04x ", pc );
    fprintf(fout, "w%02d[", m_warp_id);
    for (unsigned j=0; j<m_config->warp_size; j++)
        fprintf(fout, "%c", (active(j)?'1':'0') );
    fprintf(fout, "]: ");
    ptx_print_insn( pc, fout );
    fprintf(fout, "\n");
}

void shader_core_ctx::print_stage(unsigned int stage, FILE *fout ) const
{
   m_pipeline_reg[stage].print(fout);
   //m_pipeline_reg[stage].print(fout);
}

void shader_core_ctx::display_simt_state(FILE *fout, int mask ) const
{
    if ( (mask & 4) && m_config->model == POST_DOMINATOR ) {
       fprintf(fout,"per warp SIMT control-flow state:\n");
       unsigned n = m_config->n_thread_per_shader / m_config->warp_size;
       for (unsigned i=0; i < n; i++) {
          unsigned nactive = 0;
          for (unsigned j=0; j<m_config->warp_size; j++ ) {
             unsigned tid = i*m_config->warp_size + j;
             int done = ptx_thread_done(tid);
             nactive += (ptx_thread_done(tid)?0:1);
             if ( done && (mask & 8) ) {
                unsigned done_cycle = m_thread[tid]->donecycle();
                if ( done_cycle ) {
                   printf("\n w%02u:t%03u: done @ cycle %u", i, tid, done_cycle );
                }
             }
          }
          if ( nactive == 0 ) {
             continue;
          }
          m_simt_stack[i]->print(fout);
       }
       fprintf(fout,"\n");
    }
}


void ldst_unit::print(FILE *fout) const
{
    fprintf(fout,"LD/ST unit  = ");
    m_dispatch_reg->print(fout);
    if ( m_mem_rc != NO_RC_FAIL ) {
        fprintf(fout,"              LD/ST stall condition: ");
        switch ( m_mem_rc ) {
        case BK_CONF:        fprintf(fout,"BK_CONF"); break;
        case MSHR_RC_FAIL:   fprintf(fout,"MSHR_RC_FAIL"); break;
        case ICNT_RC_FAIL:   fprintf(fout,"ICNT_RC_FAIL"); break;
        case COAL_STALL:     fprintf(fout,"COAL_STALL"); break;
        case WB_ICNT_RC_FAIL: fprintf(fout,"WB_ICNT_RC_FAIL"); break;
        case WB_CACHE_RSRV_FAIL: fprintf(fout,"WB_CACHE_RSRV_FAIL"); break;
        case N_MEM_STAGE_STALL_TYPE: fprintf(fout,"N_MEM_STAGE_STALL_TYPE"); break;
        default: abort();
        }
        fprintf(fout,"\n");
    }
    fprintf(fout,"LD/ST wb    = ");
    m_next_wb.print(fout);
    fprintf(fout, "Last LD/ST writeback @ %llu + %llu (gpu_sim_cycle+gpu_tot_sim_cycle)\n",
                  m_last_inst_gpu_sim_cycle, m_last_inst_gpu_tot_sim_cycle );
    fprintf(fout,"Pending register writes:\n");
    std::map<unsigned/*warp_id*/, std::map<unsigned/*regnum*/,unsigned/*count*/> >::const_iterator w;
    for( w=m_pending_writes.begin(); w!=m_pending_writes.end(); w++ ) {
        unsigned warp_id = w->first;
        const std::map<unsigned/*regnum*/,unsigned/*count*/> &warp_info = w->second;
        if( warp_info.empty() ) 
            continue;
        fprintf(fout,"  w%2u : ", warp_id );
        std::map<unsigned/*regnum*/,unsigned/*count*/>::const_iterator r;
        for( r=warp_info.begin(); r!=warp_info.end(); ++r ) {
            fprintf(fout,"  %u(%u)", r->first, r->second );
        }
        fprintf(fout,"\n");
    }
    m_L1C->display_state(fout);
    m_L1T->display_state(fout);
    if( !m_config->m_L1D_config.disabled() )
    	m_L1D->display_state(fout);
    fprintf(fout,"LD/ST response FIFO (occupancy = %zu):\n", m_response_fifo.size() );
    for( std::list<mem_fetch*>::const_iterator i=m_response_fifo.begin(); i != m_response_fifo.end(); i++ ) {
        const mem_fetch *mf = *i;
        mf->print(fout);
    }
}

void shader_core_ctx::display_pipeline(FILE *fout, int print_mem, int mask ) const
{
   fprintf(fout, "=================================================\n");
   fprintf(fout, "shader %u at cycle %Lu+%Lu (%u threads running)\n", m_sid, 
           gpu_tot_sim_cycle, gpu_sim_cycle, m_not_completed);
   fprintf(fout, "=================================================\n");

   dump_warp_state(fout);
   fprintf(fout,"\n");

   m_L1I->display_state(fout);

   fprintf(fout, "IF/ID       = ");
   if( !m_inst_fetch_buffer.m_valid )
       fprintf(fout,"bubble\n");
   else {
       fprintf(fout,"w%2u : pc = 0x%x, nbytes = %u\n", 
               m_inst_fetch_buffer.m_warp_id,
               m_inst_fetch_buffer.m_pc, 
               m_inst_fetch_buffer.m_nbytes );
   }
   fprintf(fout,"\nibuffer status:\n");
   for( unsigned i=0; i<m_config->max_warps_per_shader; i++) {
       if( !m_warp[i]->ibuffer_empty() )
           m_warp[i]->print_ibuffer(fout);
   }
   fprintf(fout,"\n");
   display_simt_state(fout,mask);
   fprintf(fout, "-------------------------- Scoreboard\n");
   m_scoreboard->printContents();
/*
   fprintf(fout,"ID/OC (SP)  = ");
   print_stage(ID_OC_SP, fout);
   fprintf(fout,"ID/OC (SFU) = ");
   print_stage(ID_OC_SFU, fout);
   fprintf(fout,"ID/OC (MEM) = ");
   print_stage(ID_OC_MEM, fout);
*/
   fprintf(fout, "-------------------------- OP COL\n");
   m_operand_collector.dump(fout);
/* fprintf(fout, "OC/EX (SP)  = ");
   print_stage(OC_EX_SP, fout);
   fprintf(fout, "OC/EX (SFU) = ");
   print_stage(OC_EX_SFU, fout);
   fprintf(fout, "OC/EX (MEM) = ");
   print_stage(OC_EX_MEM, fout);
*/
   fprintf(fout, "-------------------------- Pipe Regs\n");

   for (unsigned i = 0; i < N_PIPELINE_STAGES; i++) {
       fprintf(fout,"--- %s ---\n",pipeline_stage_name_decode[i]);
       print_stage(i,fout);fprintf(fout,"\n");
   }

   fprintf(fout, "-------------------------- Fu\n");
   for( unsigned n=0; n < m_num_function_units; n++ ){
       m_fu[n]->print(fout);
       fprintf(fout, "---------------\n");
   }
   fprintf(fout, "-------------------------- other:\n");

   for(int i=0; i<num_result_bus; i++){
	   std::string bits = m_result_bus[i]->to_string();
	   fprintf(fout, "EX/WB sched[%d]= %s\n", i, bits.c_str() );
   }
   fprintf(fout, "EX/WB      = ");
   print_stage(EX_WB, fout);
   fprintf(fout, "\n");
   fprintf(fout, "Last EX/WB writeback @ %llu + %llu (gpu_sim_cycle+gpu_tot_sim_cycle)\n",
                 m_last_inst_gpu_sim_cycle, m_last_inst_gpu_tot_sim_cycle );

   if( m_active_threads.count() <= 2*m_config->warp_size ) {
       fprintf(fout,"Active Threads : ");
       unsigned last_warp_id = -1;
       for(unsigned tid=0; tid < m_active_threads.size(); tid++ ) {
           unsigned warp_id = tid/m_config->warp_size;
           if( m_active_threads.test(tid) ) {
               if( warp_id != last_warp_id ) {
                   fprintf(fout,"\n  warp %u : ", warp_id );
                   last_warp_id=warp_id;
               }
               fprintf(fout,"%u ", tid );
           }
       }
   }

}

unsigned int shader_core_config::max_cta( const kernel_info_t &k ) const
{
   unsigned threads_per_cta  = k.threads_per_cta();
   const class function_info *kernel = k.entry();
   unsigned int padded_cta_size = threads_per_cta;
   if (padded_cta_size%warp_size) 
      padded_cta_size = ((padded_cta_size/warp_size)+1)*(warp_size);

   //Limit by n_threads/shader
   unsigned int result_thread = n_thread_per_shader / padded_cta_size;

   const struct gpgpu_ptx_sim_kernel_info *kernel_info = ptx_sim_kernel_info(kernel);

   //Limit by shmem/shader
   unsigned int result_shmem = (unsigned)-1;
   if (kernel_info->smem > 0)
      result_shmem = gpgpu_shmem_size / kernel_info->smem;

   //Limit by register count, rounded up to multiple of 4.
   unsigned int result_regs = (unsigned)-1;
   if (kernel_info->regs > 0)
      result_regs = gpgpu_shader_registers / (padded_cta_size * ((kernel_info->regs+3)&~3));

   //Limit by CTA
   unsigned int result_cta = max_cta_per_core;

   unsigned result = result_thread;
   result = gs_min2(result, result_shmem);
   result = gs_min2(result, result_regs);
   result = gs_min2(result, result_cta);

   static const struct gpgpu_ptx_sim_kernel_info* last_kinfo = NULL;
   if (last_kinfo != kernel_info) {   //Only print out stats if kernel_info struct changes
      last_kinfo = kernel_info;
      printf ("GPGPU-Sim uArch: CTA/core = %u, limited by:", result);
      if (result == result_thread) printf (" threads");
      if (result == result_shmem) printf (" shmem");
      if (result == result_regs) printf (" regs");
      if (result == result_cta) printf (" cta_limit");
      printf ("\n");
   }

    //gpu_max_cta_per_shader is limited by number of CTAs if not enough to keep all cores busy    
    if( k.num_blocks() < result*num_shader() ) { 
       result = k.num_blocks() / num_shader();
       if (k.num_blocks() % num_shader())
          result++;
    }

    assert( result <= MAX_CTA_PER_SHADER );
    if (result < 1) {
       printf ("GPGPU-Sim uArch: ERROR ** Kernel requires more resources than shader has.\n");
       abort();
    }

    return result;
}

void shader_core_ctx::cycle()
{
    writeback();
    execute();
    read_operands();
    issue();
    decode();
    fetch();
}

// Flushes all content of the cache to memory

void shader_core_ctx::cache_flush()
{
   m_ldst_unit->flush();
}

// modifiers
std::list<opndcoll_rfu_t::op_t> opndcoll_rfu_t::arbiter_t::allocate_reads() 
{
   std::list<op_t> result;  // a list of registers that (a) are in different register banks, (b) do not go to the same operand collector

   int input;
   int output;
   int _inputs = m_num_banks;
   int _outputs = m_num_collectors;
   int _square = ( _inputs > _outputs ) ? _inputs : _outputs;
   assert(_square > 0);
   int _pri = (int)m_last_cu;

   // Clear matching
   for ( int i = 0; i < _inputs; ++i ) 
      _inmatch[i] = -1;
   for ( int j = 0; j < _outputs; ++j ) 
      _outmatch[j] = -1;

   for( unsigned i=0; i<m_num_banks; i++) {
      for( unsigned j=0; j<m_num_collectors; j++) {
         assert( i < (unsigned)_inputs );
         assert( j < (unsigned)_outputs );
         _request[i][j] = 0;
      }
      if( !m_queue[i].empty() ) {
         const op_t &op = m_queue[i].front();
         int oc_id = op.get_oc_id();
         assert( i < (unsigned)_inputs );
         assert( oc_id < _outputs );
         _request[i][oc_id] = 1;
      }
      if( m_allocated_bank[i].is_write() ) {
         assert( i < (unsigned)_inputs );
         _inmatch[i] = 0; // write gets priority
      }
   }

   ///// wavefront allocator from booksim... --->
   
   // Loop through diagonals of request matrix

   for ( int p = 0; p < _square; ++p ) {
      output = ( _pri + p ) % _square;

      // Step through the current diagonal
      for ( input = 0; input < _inputs; ++input ) {
          assert( input < _inputs );
          assert( output < _outputs );
         if ( ( output < _outputs ) && 
              ( _inmatch[input] == -1 ) && 
              ( _outmatch[output] == -1 ) &&
              ( _request[input][output]/*.label != -1*/ ) ) {
            // Grant!
            _inmatch[input] = output;
            _outmatch[output] = input;
         }

         output = ( output + 1 ) % _square;
      }
   }

   // Round-robin the priority diagonal
   _pri = ( _pri + 1 ) % _square;

   /// <--- end code from booksim

   m_last_cu = _pri;
   for( unsigned i=0; i < m_num_banks; i++ ) {
      if( _inmatch[i] != -1 ) {
         if( !m_allocated_bank[i].is_write() ) {
            unsigned bank = (unsigned)i;
            op_t &op = m_queue[bank].front();
            result.push_back(op);
            m_queue[bank].pop_front();
         }
      }
   }

   return result;
}

barrier_set_t::barrier_set_t( unsigned max_warps_per_core, unsigned max_cta_per_core )
{
   m_max_warps_per_core = max_warps_per_core;
   m_max_cta_per_core = max_cta_per_core;
   if( max_warps_per_core > WARP_PER_CTA_MAX ) {
      printf("ERROR ** increase WARP_PER_CTA_MAX in shader.h from %u to >= %u or warps per cta in gpgpusim.config\n",
             WARP_PER_CTA_MAX, max_warps_per_core );
      exit(1);
   }
   m_warp_active.reset();
   m_warp_at_barrier.reset();
}

// during cta allocation
void barrier_set_t::allocate_barrier( unsigned cta_id, warp_set_t warps )
{
   assert( cta_id < m_max_cta_per_core );
   cta_to_warp_t::iterator w=m_cta_to_warps.find(cta_id);
   assert( w == m_cta_to_warps.end() ); // cta should not already be active or allocated barrier resources
   m_cta_to_warps[cta_id] = warps;
   assert( m_cta_to_warps.size() <= m_max_cta_per_core ); // catch cta's that were not properly deallocated
  
   m_warp_active |= warps;
   m_warp_at_barrier &= ~warps;
}

// during cta deallocation
void barrier_set_t::deallocate_barrier( unsigned cta_id )
{
   cta_to_warp_t::iterator w=m_cta_to_warps.find(cta_id);
   if( w == m_cta_to_warps.end() )
      return;
   warp_set_t warps = w->second;
   warp_set_t at_barrier = warps & m_warp_at_barrier;
   assert( at_barrier.any() == false ); // no warps stuck at barrier
   warp_set_t active = warps & m_warp_active;
   assert( active.any() == false ); // no warps in CTA still running
   m_warp_active &= ~warps;
   m_warp_at_barrier &= ~warps;
   m_cta_to_warps.erase(w);
}

// individual warp hits barrier
void barrier_set_t::warp_reaches_barrier( unsigned cta_id, unsigned warp_id )
{
   cta_to_warp_t::iterator w=m_cta_to_warps.find(cta_id);

   if( w == m_cta_to_warps.end() ) { // cta is active
      printf("ERROR ** cta_id %u not found in barrier set on cycle %llu+%llu...\n", cta_id, gpu_tot_sim_cycle, gpu_sim_cycle );
      dump();
      abort();
   }
   assert( w->second.test(warp_id) == true ); // warp is in cta

   m_warp_at_barrier.set(warp_id);

   warp_set_t warps_in_cta = w->second;
   warp_set_t at_barrier = warps_in_cta & m_warp_at_barrier;
   warp_set_t active = warps_in_cta & m_warp_active;

   if( at_barrier == active ) {
      // all warps have reached barrier, so release waiting warps...
      m_warp_at_barrier &= ~at_barrier;
   }
}

// fetching a warp
bool barrier_set_t::available_for_fetch( unsigned warp_id ) const
{
   return m_warp_active.test(warp_id) && m_warp_at_barrier.test(warp_id);
}

// warp reaches exit 
void barrier_set_t::warp_exit( unsigned warp_id )
{
   // caller needs to verify all threads in warp are done, e.g., by checking PDOM stack to 
   // see it has only one entry during exit_impl()
   m_warp_active.reset(warp_id);

   // test for barrier release 
   cta_to_warp_t::iterator w=m_cta_to_warps.begin(); 
   for (; w != m_cta_to_warps.end(); ++w) {
      if (w->second.test(warp_id) == true) break; 
   }
   warp_set_t warps_in_cta = w->second;
   warp_set_t at_barrier = warps_in_cta & m_warp_at_barrier;
   warp_set_t active = warps_in_cta & m_warp_active;

   if( at_barrier == active ) {
      // all warps have reached barrier, so release waiting warps...
      m_warp_at_barrier &= ~at_barrier;
   }
}

// assertions
bool barrier_set_t::warp_waiting_at_barrier( unsigned warp_id ) const
{ 
   return m_warp_at_barrier.test(warp_id);
}

void barrier_set_t::dump() const
{
   printf( "barrier set information\n");
   printf( "  m_max_cta_per_core = %u\n",  m_max_cta_per_core );
   printf( "  m_max_warps_per_core = %u\n", m_max_warps_per_core );
   printf( "  cta_to_warps:\n");
   
   cta_to_warp_t::const_iterator i;
   for( i=m_cta_to_warps.begin(); i!=m_cta_to_warps.end(); i++ ) {
      unsigned cta_id = i->first;
      warp_set_t warps = i->second;
      printf("    cta_id %u : %s\n", cta_id, warps.to_string().c_str() );
   }
   printf("  warp_active: %s\n", m_warp_active.to_string().c_str() );
   printf("  warp_at_barrier: %s\n", m_warp_at_barrier.to_string().c_str() );
   fflush(stdout); 
}

void shader_core_ctx::warp_exit( unsigned warp_id )
{
	bool done = true;
	for (	unsigned i = warp_id*get_config()->warp_size;
			i < (warp_id+1)*get_config()->warp_size;
			i++ ) {

//		if(this->m_thread[i]->m_functional_model_thread_state && this->m_thread[i].m_functional_model_thread_state->donecycle()==0) {
//			done = false;
//		}


		if (m_thread[i] && !m_thread[i]->is_done()) done = false;
	}
	//if (m_warp[warp_id].get_n_completed() == get_config()->warp_size)
	//if (this->m_simt_stack[warp_id]->get_num_entries() == 0)
	if (done)
		m_barriers.warp_exit( warp_id );
}

bool shader_core_ctx::warp_waiting_at_barrier( unsigned warp_id ) const
{
   return m_barriers.warp_waiting_at_barrier(warp_id);
}

bool shader_core_ctx::warp_waiting_at_mem_barrier( unsigned warp_id ) 
{
   if( !m_warp[warp_id]->get_membar() )
      return false;
   if( !m_scoreboard->pendingWrites(warp_id) ) {
      m_warp[warp_id]->clear_membar();
      return false;
   }
   return true;
}

void shader_core_ctx::set_max_cta( const kernel_info_t &kernel ) 
{
    // calculate the max cta count and cta size for local memory address mapping
    kernel_max_cta_per_shader = m_config->max_cta(kernel);
    unsigned int gpu_cta_size = kernel.threads_per_cta();
    kernel_padded_threads_per_cta = (gpu_cta_size%m_config->warp_size) ? 
        m_config->warp_size*((gpu_cta_size/m_config->warp_size)+1) : 
        gpu_cta_size;
}

void shader_core_ctx::decrement_atomic_count( unsigned wid, unsigned n )
{
   assert( m_warp[wid]->get_n_atomic() >= n );
   m_warp[wid]->dec_n_atomic(n);
}


bool shader_core_ctx::fetch_unit_response_buffer_full() const
{
    return false;
}

void shader_core_ctx::accept_fetch_response( mem_fetch *mf )
{
    mf->set_status(IN_SHADER_FETCHED,gpu_sim_cycle+gpu_tot_sim_cycle);
    m_L1I->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);
}

bool shader_core_ctx::ldst_unit_response_buffer_full() const
{
    return m_ldst_unit->response_buffer_full();
}

void shader_core_ctx::accept_ldst_unit_response(mem_fetch * mf) 
{
   m_ldst_unit->fill(mf);
}

void shader_core_ctx::store_ack( class mem_fetch *mf )
{
	assert( mf->get_type() == WRITE_ACK  || ( m_config->gpgpu_perfect_mem && mf->get_is_write() ) );
    unsigned warp_id = mf->get_wid();
    m_warp[warp_id]->dec_store_req();
}

void shader_core_ctx::print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses ) {
   m_ldst_unit->print_cache_stats( fp, dl1_accesses, dl1_misses );
   if ( 0 == m_sid ) {
       m_scheduling_point_system.print();
   }
}

bool base_shd_warp_t::functional_done() const
{
    return get_n_completed() == m_warp_size;
}

bool base_shd_warp_t::hardware_done() const
{
    return functional_done() && stores_done() && !inst_in_pipeline(); 
}

bool base_shd_warp_t::waiting()
{
    if ( functional_done() ) {
        // waiting to be initialized with a kernel
        return true;
    } else if ( m_shader->warp_waiting_at_barrier(m_warp_id) ) {
        // waiting for other warps in CTA to reach barrier
        return true;
    } else if ( m_shader->warp_waiting_at_mem_barrier(m_warp_id) ) {
        // waiting for memory barrier
        return true;
    } else if ( m_n_atomic >0 ) {
        // waiting for atomic operation to complete at memory:
        // this stall is not required for accurate timing model, but rather we
        // stall here since if a call/return instruction occurs in the meantime
        // the functional execution of the atomic when it hits DRAM can cause
        // the wrong register to be read.
        return true;
    }
    return false;
}

void base_shd_warp_t::print( FILE *fout ) const
{
    if( !done_exit() ) {
        fprintf( fout, "w%02u npc: 0x%04x, done:%c%c%c%c:%2u i:%u iss:%u s:%u a:%u (done: ",
                m_warp_id,
                m_next_pc,
                (functional_done()?'f':' '),
                (stores_done()?'s':' '),
                (inst_in_pipeline()?' ':'i'),
                (done_exit()?'e':' '),
                n_completed,
                m_inst_in_pipeline,
                m_issued_inst,
                m_stores_outstanding,
                m_n_atomic );
        for (unsigned i = m_warp_id*m_warp_size; i < (m_warp_id+1)*m_warp_size; i++ ) {
          if ( m_shader->ptx_thread_done(i) ) fprintf(fout,"1");
          else fprintf(fout,"0");
          if ( (((i+1)%4) == 0) && (i+1) < (m_warp_id+1)*m_warp_size ) 
             fprintf(fout,",");
        }
        fprintf(fout,") ");
        fprintf(fout," active=%s", m_active_threads.to_string().c_str() );
        fprintf(fout," last fetched @ %5llu", m_last_fetch);
        if( m_imiss_pending ) 
            fprintf(fout," i-miss pending");
        fprintf(fout,"\n");
    }
}

void base_shd_warp_t::print_ibuffer( FILE *fout ) const
{
    fprintf(fout,"  ibuffer[%2u] : ", m_warp_id );
    for( unsigned i=0; i < IBUFFER_SIZE; i++) {
        const inst_t *inst = m_ibuffer[i]->m_inst;
        if( inst ) inst->print_insn(fout);
        else if( m_ibuffer[i]->m_valid )
           fprintf(fout," <invalid instruction> ");
        else fprintf(fout," <empty> ");
    }
    fprintf(fout,"\n");
}

void base_shd_warp_t::ibuffer_flush()
{
	for(unsigned i=0;i<IBUFFER_SIZE;i++) {
		if( m_ibuffer[i]->m_valid )
			dec_inst_in_pipeline();
		m_ibuffer[i]->m_inst=NULL;
		m_ibuffer[i]->m_valid=false;
	}
}

void base_shd_warp_t::ibuffer_fill( unsigned slot, const warp_inst_t *pI ) {
   assert(!m_ibuffer[slot]->m_valid);
   assert(slot < IBUFFER_SIZE );
   m_ibuffer[slot]->m_inst=pI;
   m_ibuffer[slot]->m_valid=true;
   m_next=0;
}

void opndcoll_rfu_t::add_cu_set(unsigned set_id, unsigned num_cu, unsigned num_dispatch){
    m_cus[set_id].reserve(num_cu); //this is necessary to stop pointers in m_cu from being invalid do to a resize;
    for (unsigned i = 0; i < num_cu; i++) {
        m_cus[set_id].push_back(collector_unit_t());
        m_cu.push_back(&m_cus[set_id].back());
    }
    // for now each collector set gets dedicated dispatch units.
    for (unsigned i = 0; i < num_dispatch; i++) {
        m_dispatch_units.push_back(dispatch_unit_t(&m_cus[set_id]));
    }
}


void opndcoll_rfu_t::add_port(port_vector_t & input, port_vector_t & output, uint_vector_t cu_sets)
{
    //m_num_ports++;
    //m_num_collectors += num_collector_units;
    //m_input.resize(m_num_ports);
    //m_output.resize(m_num_ports);
    //m_num_collector_units.resize(m_num_ports);
    //m_input[m_num_ports-1]=input_port;
    //m_output[m_num_ports-1]=output_port;
    //m_num_collector_units[m_num_ports-1]=num_collector_units;
    m_in_ports.push_back(input_port_t(input,output,cu_sets));
}

void opndcoll_rfu_t::init( unsigned num_banks, shader_core_ctx *shader )
{
   m_shader=shader;
   m_arbiter.init(m_cu.size(),num_banks);
   //for( unsigned n=0; n<m_num_ports;n++ ) 
   //    m_dispatch_units[m_output[n]].init( m_num_collector_units[n] );
   m_num_banks = num_banks;
   m_bank_warp_shift = 0; 
   m_warp_size = shader->get_config()->warp_size;
   m_bank_warp_shift = (unsigned)(int) (log(m_warp_size+0.5) / log(2.0));
   assert( (m_bank_warp_shift == 5) || (m_warp_size != 32) );

   for( unsigned j=0; j<m_cu.size(); j++) {
       m_cu[j]->init(j,num_banks,m_bank_warp_shift,shader->get_config(),this);
   }
   m_initialized=true;
}

int register_bank(int regnum, int wid, unsigned num_banks, unsigned bank_warp_shift)
{
   int bank = regnum;
   if (bank_warp_shift)
      bank += wid;
   return bank % num_banks;
}

bool opndcoll_rfu_t::writeback( const warp_inst_t &inst )
{
   assert( !inst.empty() );
   std::list<unsigned> regs = m_shader->get_regs_written(inst);
   std::list<unsigned>::iterator r;
   unsigned n=0;
   for( r=regs.begin(); r!=regs.end();r++,n++ ) {
      unsigned reg = *r;
      unsigned bank = register_bank(reg,inst.warp_id(),m_num_banks,m_bank_warp_shift);
      if( m_arbiter.bank_idle(bank) ) {
          m_arbiter.allocate_bank_for_write(bank,op_t(&inst,reg,m_num_banks,m_bank_warp_shift));
      } else {
          return false;
      }
   }
   return true;
}

void opndcoll_rfu_t::dispatch_ready_cu()
{
   for( unsigned p=0; p < m_dispatch_units.size(); ++p ) {
      dispatch_unit_t &du = m_dispatch_units[p];
      collector_unit_t *cu = du.find_ready();
      if( cu ) {
         cu->dispatch();
      }
   }
}

void opndcoll_rfu_t::allocate_cu( unsigned port_num )
{
   input_port_t& inp = m_in_ports[port_num];
   for (unsigned i = 0; i < inp.m_in.size(); i++) {
       if( (*inp.m_in[i]).has_ready() ) {
          //find a free cu 
          for (unsigned j = 0; j < inp.m_cu_sets.size(); j++) {
              std::vector<collector_unit_t> & cu_set = m_cus[inp.m_cu_sets[j]];
	      bool allocated = false;
              for (unsigned k = 0; k < cu_set.size(); k++) {
                  if(cu_set[k].is_free()) {
                     collector_unit_t *cu = &cu_set[k];
                     allocated = cu->allocate(inp.m_in[i],inp.m_out[i]);
                     m_arbiter.add_read_requests(cu);
                     break;
                  }
              }
              if (allocated) break; //cu has been allocated, no need to search more.
          }
          break; // can only service a single input, if it failed it will fail for others.
       }
   }
}

void opndcoll_rfu_t::allocate_reads()
{
   // process read requests that do not have conflicts
   std::list<op_t> allocated = m_arbiter.allocate_reads();
   std::map<unsigned,op_t> read_ops;
   for( std::list<op_t>::iterator r=allocated.begin(); r!=allocated.end(); r++ ) {
      const op_t &rr = *r;
      unsigned reg = rr.get_reg();
      unsigned wid = rr.get_wid();
      unsigned bank = register_bank(reg,wid,m_num_banks,m_bank_warp_shift);
      m_arbiter.allocate_for_read(bank,rr);
      read_ops[bank] = rr;
   }
   std::map<unsigned,op_t>::iterator r;
   for(r=read_ops.begin();r!=read_ops.end();++r ) {
      op_t &op = r->second;
      unsigned cu = op.get_oc_id();
      unsigned operand = op.get_operand();
      m_cu[cu]->collect_operand(operand);
   }
} 

bool opndcoll_rfu_t::collector_unit_t::ready() const 
{ 
   return (!m_free) && m_not_ready.none() && (*m_output_register).has_free(); 
}

void opndcoll_rfu_t::collector_unit_t::dump(FILE *fp, const shader_core_ctx *shader ) const
{
   if( m_free ) {
      fprintf(fp,"    <free>\n");
   } else {
      m_warp->print(fp);
      for( unsigned i=0; i < MAX_REG_OPERANDS*2; i++ ) {
         if( m_not_ready.test(i) ) {
            std::string r = m_src_op[i].get_reg_string();
            fprintf(fp,"    '%s' not ready\n", r.c_str() );
         }
      }
   }
}

void opndcoll_rfu_t::collector_unit_t::init( unsigned n, 
                                             unsigned num_banks, 
                                             unsigned log2_warp_size,
                                             const core_config *config,
                                             opndcoll_rfu_t *rfu ) 
{ 
   m_rfu=rfu;
   m_cuid=n; 
   m_num_banks=num_banks;
   assert(m_warp==NULL); 
   m_warp = new warp_inst_t(config);
   m_bank_warp_shift=log2_warp_size;
}

bool opndcoll_rfu_t::collector_unit_t::allocate( register_set* pipeline_reg_set, register_set* output_reg_set ) 
{
   assert(m_free);
   assert(m_not_ready.none());
   m_free = false;
   m_output_register = output_reg_set;
   warp_inst_t **pipeline_reg = pipeline_reg_set->get_ready();
   if( (pipeline_reg) and !((*pipeline_reg)->empty()) ) {
      m_warp_id = (*pipeline_reg)->warp_id();
      for( unsigned op=0; op < MAX_REG_OPERANDS; op++ ) {
         int reg_num = (*pipeline_reg)->arch_reg.src[op]; // this math needs to match that used in function_info::ptx_decode_inst
         if( reg_num >= 0 ) { // valid register
            m_src_op[op] = op_t( this, op, reg_num, m_num_banks, m_bank_warp_shift );
            m_not_ready.set(op);
         } else 
            m_src_op[op] = op_t();
      }
      //move_warp(m_warp,*pipeline_reg);
      pipeline_reg_set->move_out_to(m_warp);
      return true;
   }
   return false;
}

void opndcoll_rfu_t::collector_unit_t::dispatch()
{
   assert( m_not_ready.none() );
   //move_warp(*m_output_register,m_warp);
   m_output_register->move_in(m_warp);
   m_free=true;
   m_output_register = NULL;
   for( unsigned i=0; i<MAX_REG_OPERANDS*2;i++)
      m_src_op[i].reset();
}




simt_core_cluster::simt_core_cluster( class gpgpu_sim *gpu, 
                                      unsigned cluster_id, 
                                      const struct shader_core_config *config, 
                                      const struct memory_config *mem_config,
                                      shader_core_stats *stats, 
                                      class memory_stats_t *mstats )
{
    m_config = config;
    m_cta_issue_next_core=m_config->n_simt_cores_per_cluster-1; // this causes first launch to use hw cta 0
    m_cluster_id=cluster_id;
    m_gpu = gpu;
    m_stats = stats;
    m_memory_stats = mstats;
    m_core = new shader_core_ctx*[ config->n_simt_cores_per_cluster ];
    for( unsigned i=0; i < config->n_simt_cores_per_cluster; i++ ) {
        unsigned sid = m_config->cid_to_sid(i,m_cluster_id);
        m_core[i] = new shader_core_ctx(gpu,this,sid,m_cluster_id,config,mem_config,stats);
        m_core_sim_order.push_back(i); 
    }
}

void simt_core_cluster::core_cycle()
{
    for( std::list<unsigned>::iterator it = m_core_sim_order.begin(); it != m_core_sim_order.end(); ++it ) {
        m_core[*it]->cycle();
    }

    if (m_config->simt_core_sim_order == 1) {
        m_core_sim_order.splice(m_core_sim_order.end(), m_core_sim_order, m_core_sim_order.begin()); 
    }
}

void simt_core_cluster::reinit()
{
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        m_core[i]->reinit(0,m_config->n_thread_per_shader,true);
}

unsigned simt_core_cluster::max_cta( const kernel_info_t &kernel )
{
    return m_config->n_simt_cores_per_cluster * m_config->max_cta(kernel);
}

unsigned simt_core_cluster::get_not_completed() const
{
    unsigned not_completed=0;
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        not_completed += m_core[i]->get_not_completed();
    return not_completed;
}

void simt_core_cluster::print_not_completed( FILE *fp ) const
{
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) {
        unsigned not_completed=m_core[i]->get_not_completed();
        unsigned sid=m_config->cid_to_sid(i,m_cluster_id);
        fprintf(fp,"%u(%u) ", sid, not_completed );
    }
}

unsigned simt_core_cluster::get_n_active_cta() const
{
    unsigned n=0;
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        n += m_core[i]->get_n_active_cta();
    return n;
}

unsigned simt_core_cluster::issue_block2core()
{
    unsigned num_blocks_issued=0;
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) {
        unsigned core = (i+m_cta_issue_next_core+1)%m_config->n_simt_cores_per_cluster;
        if( m_core[core]->get_not_completed() == 0 ) {
            if( m_core[core]->get_kernel() == NULL ) {
                kernel_info_t *k = m_gpu->select_kernel();
                if( k ) 
                    m_core[core]->set_kernel(k);
            }
        }
        kernel_info_t *kernel = m_core[core]->get_kernel();
        if( kernel && !kernel->no_more_ctas_to_run() && (m_core[core]->get_n_active_cta() < m_config->max_cta(*kernel)) ) {
            m_core[core]->issue_block2core(*kernel);
            num_blocks_issued++;
            m_cta_issue_next_core=core; 
            break;
        }
    }
    return num_blocks_issued;
}

void simt_core_cluster::cache_flush()
{
    for( unsigned i=0; i < m_config->n_simt_cores_per_cluster; i++ ) 
        m_core[i]->cache_flush();
}

bool simt_core_cluster::icnt_injection_buffer_full(unsigned size, bool write)
{
    unsigned request_size = size;
    if (!write) 
        request_size = READ_PACKET_SIZE;
    return ! ::icnt_has_buffer(m_cluster_id, request_size);
}

void simt_core_cluster::icnt_inject_request_packet(class mem_fetch *mf)
{
    // stats
    if (mf->get_is_write()) m_stats->made_write_mfs++;
    else m_stats->made_read_mfs++;
    switch (mf->get_access_type()) {
    case CONST_ACC_R: m_stats->gpgpu_n_mem_const++; break;
    case TEXTURE_ACC_R: m_stats->gpgpu_n_mem_texture++; break;
    case GLOBAL_ACC_R: m_stats->gpgpu_n_mem_read_global++; break;
    case GLOBAL_ACC_W: m_stats->gpgpu_n_mem_write_global++; break;
    case LOCAL_ACC_R: m_stats->gpgpu_n_mem_read_local++; break;
    case LOCAL_ACC_W: m_stats->gpgpu_n_mem_write_local++; break;
    case INST_ACC_R: m_stats->gpgpu_n_mem_read_inst++; break;
    case L1_WRBK_ACC: m_stats->gpgpu_n_mem_write_global++; break;
    case VM_POLICY_CHANGE_REQ: break;
    default: assert(0);
    }
   unsigned destination = mf->get_tlx_addr().chip;
   mf->set_status(IN_ICNT_TO_MEM,gpu_sim_cycle+gpu_tot_sim_cycle);
   if (!mf->get_is_write() && !mf->isatomic())
      ::icnt_push(m_cluster_id, m_config->mem2device(destination), (void*)mf, mf->get_ctrl_size() );
   else 
      ::icnt_push(m_cluster_id, m_config->mem2device(destination), (void*)mf, mf->size());
}

void simt_core_cluster::icnt_cycle()
{
    if( !m_response_fifo.empty() ) {
        mem_fetch *mf = m_response_fifo.front();
        unsigned cid = m_config->sid_to_cid(mf->get_sid());
        if( mf->get_access_type() == INST_ACC_R ) {
            // instruction fetch response
            if( !m_core[cid]->fetch_unit_response_buffer_full() ) {
                m_response_fifo.pop_front();
                m_core[cid]->accept_fetch_response(mf);
            }
        } else {
            // data response
            if( !m_core[cid]->ldst_unit_response_buffer_full() ) {
                m_response_fifo.pop_front();
                m_memory_stats->memlatstat_read_done(mf);
                m_core[cid]->accept_ldst_unit_response(mf);
            }
        }
    }
    if( m_response_fifo.size() < m_config->n_simt_ejection_buffer_size ) {
        mem_fetch *mf = (mem_fetch*) ::icnt_pop(m_cluster_id);
        if (!mf) 
            return;
        assert(mf->get_tpc() == m_cluster_id);
        assert(mf->get_type() == READ_REPLY || mf->get_type() == WRITE_ACK || mf->get_type() == PROTOCOL_MSG );
        mf->set_status(IN_CLUSTER_TO_SHADER_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
        //m_memory_stats->memlatstat_read_done(mf,m_shader_config->max_warps_per_shader);
        m_response_fifo.push_back(mf);
    }
}

void simt_core_cluster::get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc ) const
{
    unsigned cid = m_config->sid_to_cid(sid);
    m_core[cid]->get_pdom_stack_top_info(tid,pc,rpc);
}

void simt_core_cluster::display_pipeline( unsigned sid, FILE *fout, int print_mem, int mask )
{
    m_core[m_config->sid_to_cid(sid)]->display_pipeline(fout,print_mem,mask);

    fprintf(fout,"\n");
    fprintf(fout,"Cluster %u pipeline state\n", m_cluster_id );
    fprintf(fout,"Response FIFO (occupancy = %zu):\n", m_response_fifo.size() );
    for( std::list<mem_fetch*>::const_iterator i=m_response_fifo.begin(); i != m_response_fifo.end(); i++ ) {
        const mem_fetch *mf = *i;
        mf->print(fout);
    }
}

void simt_core_cluster::print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses ) const {
   for ( unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i ) {
      m_core[ i ]->print_cache_stats( fp, dl1_accesses, dl1_misses );
   }
}

void shader_core_ctx::checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t, unsigned tid)
{
    if( inst.has_callback(t) ) 
           m_warp[inst.warp_id()]->inc_n_atomic();
        if (inst.space.is_local() && (inst.is_load() || inst.is_store())) {
            new_addr_type localaddrs[MAX_ACCESSES_PER_INSN_PER_THREAD];
            unsigned num_addrs;
            num_addrs = translate_local_memaddr(inst.get_addr(t), tid, m_config->n_simt_clusters*m_config->n_simt_cores_per_cluster,
                   inst.data_size, (new_addr_type*) localaddrs );
            inst.set_addr(t, (new_addr_type*) localaddrs, num_addrs);
        }
        if ( ptx_thread_done(tid) ) {
            m_warp[inst.warp_id()]->set_completed(t);
            m_warp[inst.warp_id()]->ibuffer_flush();
        }

    // PC-Histogram Update 
    unsigned warp_id = inst.warp_id(); 
    unsigned pc = inst.pc; 
    for (unsigned t = 0; t < m_config->warp_size; t++) {
        if (inst.active(t)) {
            int tid = warp_id * m_config->warp_size + t; 
            // tgrogers - TODO fix this I believe this should be a new logging class that signifies instructions finishing execution
            //cflog_update_thread_pc(m_sid, tid, pc);  
        }
    }
}

void simt_core_cluster::final_eval_cache_lines(){
	for(unsigned i=0; i<m_config->n_simt_cores_per_cluster; i++){
		m_core[i]->final_eval_cache_lines();
	}
}

unsigned simt_core_cluster::get_all_never_used(){
	unsigned total = 0;
	for(unsigned i=0; i<m_config->n_simt_cores_per_cluster; i++){
		total += m_core[i]->get_never_used();
	}
	return total;
}
unsigned simt_core_cluster::get_all_used_now(){
	unsigned total = 0;
	for(unsigned i=0; i<m_config->n_simt_cores_per_cluster; i++){
		total += m_core[i]->get_used_now();
	}
	return total;
}
unsigned simt_core_cluster::get_all_event_used(){
	unsigned total = 0;
	for(unsigned i=0; i<m_config->n_simt_cores_per_cluster; i++){
		total += m_core[i]->get_event_used();
	}
	return total;
}

unsigned simt_core_cluster::get_used_cache_array(unsigned idx){
	unsigned total[8] = {0};
	for(unsigned i=0; i<m_config->n_simt_cores_per_cluster; i++){
		for(unsigned j=0; j<8; j++)
			total[j] += m_core[i]->get_used_cache_array(j);
	}
    	assert(idx < 8);
	return total[idx];
}

unsigned simt_core_cluster::get_event_used_cache_array(unsigned idx){
	unsigned total[8] = {0};
	for(unsigned i=0; i<m_config->n_simt_cores_per_cluster; i++){
		for(unsigned j=0; j<8; j++)
			total[j] += m_core[i]->get_event_used_cache_array(j);
	}
    	assert(idx < 8);
	return total[idx];
}


shader_tlb_interface::shader_tlb_interface( const char *name,
         const tlb_config &config,
         int core_id, int type_id,
         mem_fetch_interface *memport,
         enum mem_fetch_status status,
         gpgpu_sim* the_gpu)
            : read_only_cache( name, config, core_id, type_id, memport, status, false ), m_config(config), m_the_gpu( the_gpu ) {

}

shader_ideal_tlb::shader_ideal_tlb( const char *name,
         const tlb_config &config,
         int core_id, int type_id,
         mem_fetch_interface *memport,
         enum mem_fetch_status status,
         gpgpu_sim* the_gpu )
            : shader_tlb_interface( name, config, core_id, type_id, memport, status, the_gpu ) {

}

extern std::map< unsigned, unsigned > g_objects_accessed;
extern std::map< new_addr_type, unsigned > g_cache_lines_accessed;
extern size_t g_hack_start_of_static_space;
extern size_t g_hack_end_of_static_space;
cache_request_status shader_tlb_interface::translate_addrs( warp_inst_t& inst ) {
    for( unsigned i=0; i < inst.warp_size(); i++ ) {
        if( inst.active( i ) ) {
            virtual_object_divided_page* page = m_the_gpu->get_virtualized_page( inst.get_addr( i ) );
            if ( page ) {
                vm_page_mapping_payload new_map( g_translated_page_type, page );
                inst.set_vm_policy( i, new_map );
            }

            if ( page && page->get_object_size() > 0 ) {
                const size_t object_num
                    = ( inst.get_addr( i ) - g_hack_start_of_static_space ) / page->get_object_size();
                // Test
                if ( g_objects_accessed.find( object_num ) == g_objects_accessed.end() ) {
                    g_objects_accessed[ object_num ] = 1;
                } else {
                    g_objects_accessed[ object_num ]++;
                }
                if ( m_objects_accessed.find( object_num ) == m_objects_accessed.end() ) {
                    m_objects_accessed[ object_num ] = 1;
                } else {
                    m_objects_accessed[ object_num ]++;
                }
            }

            new_addr_type linear_addr = inst.get_addr( i );

            virtual_page_type translation_type = VP_NON_TRANSLATED;
            cache_request_status status = access( inst.get_addr( i ), linear_addr, translation_type );
            if ( status != HIT ) return MISS;
            inst.set_addr( i, linear_addr );

            if ( linear_addr >= g_hack_start_of_static_space && linear_addr < g_hack_end_of_static_space ) {
                new_addr_type cache_line = linear_addr / 128;
                if ( g_cache_lines_accessed.find( cache_line ) == g_cache_lines_accessed.end() ) {
                    g_cache_lines_accessed[ cache_line ] = 1;
                } else {
                    g_cache_lines_accessed[ cache_line ]++;
                }
                if ( m_cache_lines_accessed.find( cache_line ) == m_cache_lines_accessed.end() ) {
                    m_cache_lines_accessed[ cache_line ] = 1;
                } else {
                    m_cache_lines_accessed[ cache_line ]++;
                }
            }
        }
    }
    return HIT;
}

void shader_tlb_interface::print( FILE* fp ) const {
   fprintf( fp, "TLB memcached_total_objects_accessed = %zu\n", m_objects_accessed.size() );
   fprintf( fp, "TLB memcached_total_remapped_cache_lines_touched = %zu\n", m_cache_lines_accessed.size() );

   if ( DYNAMIC_TRANSLATION == g_translation_config ) {
       fprintf( fp, "Logged_Regions:\n" );
       m_the_gpu->print_virtualized_pages( fp, true );
   }
}

cache_request_status shader_ideal_tlb::access( new_addr_type virtual_mem_addr, new_addr_type& linear_mem_addr, virtual_page_type& page_translation )
{
    page_translation = VP_NON_TRANSLATED;
    linear_mem_addr = virtual_mem_addr;
    return HIT;
}
/*
cache_request_status shader_dynamic_vm_manager_tlb::translate_addrs( warp_inst_t& inst ) {
   for( unsigned i=0; i < inst.warp_size(); i++ ) {
      if( inst.active( i ) ) {
         virtual_object_divided_page* page = m_the_gpu->get_virtualized_page( inst.get_addr( i ) );
         if ( page ) {
            vm_page_mapping_payload new_map( g_translated_page_type, page );
            inst.set_vm_policy( i, &new_map );
         } else {
            vm_page_mapping_payload new_map;
            inst.set_vm_policy( i, &new_map );
         }

         new_addr_type linear_addr = inst.get_addr( i );
         virtual_page_type page_translation = VP_NON_TRANSLATED;
         cache_request_status status = access( inst.get_addr( i ), linear_addr, page_translation );
         if ( status != HIT ) 
             return MISS;
         if( page_translation == VP_NON_TRANSLATED ) {
             // either not dynamic vm, or dynamic vm and this access is to a byte in the cold region
             //assert( m_config.get_translation_config() != STATIC_TILED_TRANSLATION || inst.get_vm_policy
             vm_page_mapping_payload new_map;
             inst.set_vm_policy( i, &new_map );
         }
         inst.set_addr( i, linear_addr );

         if( page && inst.get_addr( i ) >= g_hack_start_of_static_space && inst.get_addr( i ) < g_hack_end_of_static_space ) {
            new_addr_type cache_line = linear_addr / 128;
            if ( g_cache_lines_accessed.find( cache_line ) == g_cache_lines_accessed.end() ) {
              g_cache_lines_accessed[ cache_line ] = 1;
            } else {
              g_cache_lines_accessed[ cache_line ]++;
            }
            if ( m_cache_lines_accessed.find( cache_line ) == m_cache_lines_accessed.end() ) {
              m_cache_lines_accessed[ cache_line ] = 1;
            } else {
              m_cache_lines_accessed[ cache_line ]++;
            }
         }
      }
   }
   return HIT;
}
*/
shader_page_reordering_tlb::shader_page_reordering_tlb( const char *name,
         const tlb_config &config,
         int core_id, int type_id,
         mem_fetch_interface *memport,
         enum mem_fetch_status status,
         gpgpu_sim* the_gpu,
         const vm_policy_manager* policy_man,
         virtual_policy_cache* dl1 )
            : shader_tlb_interface( name, config, core_id, type_id, memport, status, the_gpu ),
              m_policy_manager( policy_man ), m_current_state( NOT_FLUSHING_L1 ), m_dl1_cache( dl1 ){

}

cache_request_status shader_page_reordering_tlb::access( new_addr_type virtual_mem_addr, new_addr_type& linear_mem_addr, virtual_page_type& page_translation )
{
    page_translation = VP_NON_TRANSLATED;
    const virtual_object_divided_page* page = m_the_gpu->get_virtualized_page( virtual_mem_addr );
    if( NO_TRANSLATION != m_config.get_translation_config() ) {
        if ( page ) {
            linear_mem_addr = page->translate( virtual_mem_addr );
            page_translation = g_translated_page_type;
        }
    }
    return HIT;
}

cache_request_status shader_page_reordering_tlb::access( new_addr_type virtual_mem_addr, base_virtual_page*& policy_used ) const {
   // This is the point where we should be checking our tlb storage space to see if we have cached the region.
   // For now we are going to assume an infinite amount of tlb storage space that is shared between all the cores
   policy_used = dynamic_cast< base_virtual_page* >( m_the_gpu->get_virtualized_page( virtual_mem_addr ) );
   return HIT;
}

void shader_page_reordering_tlb::insert_new_policy( const vm_page_mapping_payload& policy ) {
    // The current assumption is that everything is mapped as default unless it is in our region list.
    // Since the policy manager can change it's mind on a region we also need to delete and change regions.
    const new_addr_type start_addr = ( new_addr_type ) policy.get_virtualized_page()->get_start_addr();
    virtual_object_divided_page* page = dynamic_cast< virtual_object_divided_page* >( policy.get_virtualized_page() );

    if ( !page ) {
        if ( m_the_gpu->get_virtualized_page( start_addr ) ) {
            m_the_gpu->remove_virtualized_page( start_addr );
            m_stats.remove_page_event();
        }
    } else if ( m_config.get_virtual_page_type() == VP_TILED ) {
        size_t total_region_size = (new_addr_type)page->get_end_addr() - (new_addr_type)page->get_start_addr();
        virtual_tiled_page* tiled_page = dynamic_cast< virtual_tiled_page* >( page );
        if ( page && total_region_size % ( tiled_page->get_tile_width() * tiled_page->get_object_size())  != 0 ) {
            fprintf(stderr, "ERROR shader_dynamic_vm_manager_tlb::insert_new_policy: Virtual memory space requires a tile aligned region space, the total_region_size=%zu was passed in.  Please ensure a multiple of %zu\n"
                    , total_region_size, tiled_page->get_tile_width() * tiled_page->get_object_size() );
            abort();
        }
        m_the_gpu->set_dynamic_virtualized_page( page );
    } else if ( m_config.get_virtual_page_type() == VP_HOT_DATA_ALIGNED ) {
        size_t total_region_size = (new_addr_type)page->get_end_addr() - (new_addr_type)page->get_start_addr();
        virtual_hot_page* tiled_page = dynamic_cast< virtual_hot_page* >( page );
        if ( page && total_region_size % ( tiled_page->get_object_size())  != 0 ) {
            fprintf(stderr, "ERROR shader_dynamic_vm_manager_tlb::insert_new_policy: Virtual memory space requires a tile aligned region space, the total_region_size=%zu was passed in.  Please ensure a multiple of %zu\n"
                    , total_region_size, tiled_page->get_object_size() );
            abort();
        }
        m_the_gpu->set_dynamic_virtualized_page( page );
    } else if ( m_config.get_virtual_page_type() == VP_HOT_DATA_UNALIGNED ) {
        m_the_gpu->set_dynamic_virtualized_page( page );
    } else {
        printf("shader_dynamic_vm_manager_tlb::insert_new_policy - Unknown translation\n");
        abort();
    }
    // We need to start flushing all the L1 lines that don't share this sense of the world
    // TODO re-enable the flush
    //flush_invalid_L1_lines( policy );
}

void shader_page_reordering_tlb::flush_invalid_L1_lines( const vm_page_mapping_payload& policy ) {
   // Go through all the cache lines that this translation effects and evict them if they do not match the new policy
   // This process should take some time, and we need to inform the policy manager when we are done
   policy_flush_entry entry;
   entry.mapping = policy;
   entry.next_line_to_flushed = (new_addr_type)policy.get_virtualized_page()->get_start_addr();
   // this code assumes a cache line aligned region space.
   assert( entry.next_line_to_flushed % m_dl1_cache->get_line_sz() == 0 );
   m_flush_entries.push_back( entry );
   m_current_state = FLUSHING_L1;
}

void shader_page_reordering_tlb::cycle() {
   // Lets flush everything all at once for simplicity
   if ( FLUSHING_L1 == m_current_state ) {
      std::list< policy_flush_entry >::iterator it = m_flush_entries.begin();
      for ( ; it != m_flush_entries.end(); ++it ) {
         while( it->next_line_to_flushed < (new_addr_type)it->mapping.get_virtualized_page()->get_end_addr() ) {
            m_dl1_cache->invalidate_line_if_policy_does_not_match( it->next_line_to_flushed, it->mapping );
            it->next_line_to_flushed += m_dl1_cache->get_line_sz();
         }
      }
      m_flush_entries.clear();
      m_current_state = NOT_FLUSHING_L1;
      // TODO inform the vm man that the flush is done
   }
}
