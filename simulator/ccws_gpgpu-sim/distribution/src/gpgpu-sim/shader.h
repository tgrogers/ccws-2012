// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Andrew Turner,
// Ali Bakhoda 
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

#ifndef SHADER_H
#define SHADER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <map>
#include <set>
#include <vector>
#include <list>
#include <bitset>
#include <utility>
#include <algorithm>
#include <deque>

#include "../cuda-sim/ptx.tab.h"

#include "delayqueue.h"
#include "stack.h"
#include "dram.h"
#include "../abstract_hardware_model.h"
#include "scoreboard.h"
#include "mem_fetch.h"
#include "stats.h"
#include "gpu-cache.h"
#include "protected-cache.h"
#include "distil-cache.h"
#include "virtual-policy-cache.h"
#include "scheduling-point-system.h"

#define NO_OP_FLAG            0xFF

/* READ_PACKET_SIZE:
   bytes: 6 address (flit can specify chanel so this gives up to ~2GB/channel, so good for now),
          2 bytes   [shaderid + mshrid](14 bits) + req_size(0-2 bits if req_size variable) - so up to 2^14 = 16384 mshr total 
 */

#define READ_PACKET_SIZE 8

//WRITE_PACKET_SIZE: bytes: 6 address, 2 miscelaneous. 
#define WRITE_PACKET_SIZE 8

#define WRITE_MASK_SIZE 8

//Set a hard limit of 32 CTAs per shader [cuda only has 8]
#define MAX_CTA_PER_SHADER 32

class print_streams;
class dynamic_per_pc_stats;

enum shader_warp_ifetch_order {
    SHADER_WARP_IFETCH_ORDER_RR = 0,
    SHADER_WARP_IFETCH_ORDER_GTO
};

class thread_ctx_t {
public:
   unsigned m_cta_id; // hardware CTA this thread belongs

   // per thread stats (ac stands for accumulative).
   unsigned n_insn;
   unsigned n_insn_ac;
   unsigned n_l1_mis_ac;
   unsigned n_l1_mrghit_ac;
   unsigned n_l1_access_ac; 

   bool m_active; 
};

class base_shd_warp_t {
public:
    virtual ~base_shd_warp_t(){}
    void reset()
    {
        assert( m_stores_outstanding==0);
        assert( m_inst_in_pipeline==0);
        m_imiss_pending=false;
        m_warp_id=(unsigned)-1;
        n_completed = m_warp_size; 
        m_n_atomic=0;
        m_membar=false;
        m_done_exit=true;
        m_last_fetch=0;
        m_next=0;
        m_issued_inst = 0;
    }
    void init( address_type start_pc, unsigned cta_id, unsigned wid, const std::bitset<MAX_WARP_SIZE> &active, unsigned dynamic_warp_id )
    {
        m_cta_id=cta_id;
        m_warp_id=wid;
        m_dynamic_warp_id=dynamic_warp_id;
        m_next_pc=start_pc;
        assert( n_completed >= active.count() );
        assert( n_completed <= m_warp_size);
        n_completed   -= active.count(); // active threads are not yet completed
        m_active_threads = active;
        m_done_exit=false;
    }

    bool functional_done() const;
    bool waiting(); // not const due to membar
    bool hardware_done() const;

    bool done_exit() const { return m_done_exit; }
    void set_done_exit() { m_done_exit=true; }

    void print( FILE *fout ) const;
    void print_ibuffer( FILE *fout ) const;

    unsigned get_n_completed() const { return n_completed; }
    void set_completed( unsigned lane ) 
    { 
        assert( m_active_threads.test(lane) );
        m_active_threads.reset(lane);
        n_completed++; 
    }

    void set_last_fetch( unsigned long long sim_cycle ) { m_last_fetch=sim_cycle; }

    unsigned get_n_atomic() const { return m_n_atomic; }
    void inc_n_atomic() { m_n_atomic++; }
    void dec_n_atomic(unsigned n) { m_n_atomic-=n; }

    void set_membar() { m_membar=true; }
    void clear_membar() { m_membar=false; }
    bool get_membar() const { return m_membar; }
    address_type get_pc() const { return m_next_pc; }
    void set_next_pc( address_type pc ) { m_next_pc = pc; }

    virtual void ibuffer_fill( unsigned slot, const warp_inst_t *pI );
    bool ibuffer_empty() const
    {
        for( unsigned i=0; i < IBUFFER_SIZE; i++) 
            if(m_ibuffer[i]->m_valid)
                return false;
        return true;
    }
    virtual void ibuffer_flush();
    const warp_inst_t *ibuffer_next_inst() { return m_ibuffer[m_next]->m_inst; }
    bool ibuffer_next_valid() { return m_ibuffer[m_next]->m_valid; }
    virtual void ibuffer_free()
    {
        m_ibuffer[m_next]->m_inst = NULL;
        m_ibuffer[m_next]->m_valid = false;
    }
    void ibuffer_step() { m_next = (m_next+1)%IBUFFER_SIZE; }

    bool imiss_pending() const { return m_imiss_pending; }
    void set_imiss_pending() { m_imiss_pending=true; }
    void clear_imiss_pending() { m_imiss_pending=false; }

    bool stores_done() const { return m_stores_outstanding == 0; }
    void inc_store_req() { m_stores_outstanding++; }
    void dec_store_req() 
    {
        assert( m_stores_outstanding > 0 );
        m_stores_outstanding--;
    }

    bool inst_in_pipeline() const { return m_inst_in_pipeline > 0; }
    void inc_inst_in_pipeline() { m_inst_in_pipeline++; }
    void dec_inst_in_pipeline() 
    {
        assert( m_inst_in_pipeline > 0 );
        m_inst_in_pipeline--;
    }

    void inc_inst_issued( unsigned num_issued = 1 ) { m_issued_inst += num_issued; }

    unsigned get_cta_id() const { return m_cta_id; }

    size_t get_active_threads() const { return m_active_threads.count(); }
    unsigned get_dynamic_warp_id() const { return m_dynamic_warp_id; }
    unsigned get_warp_id() const { return m_warp_id; }

protected:
    base_shd_warp_t( class shader_core_ctx *shader, unsigned warp_size )
        : m_shader(shader), m_warp_size(warp_size), m_ibuffer( IBUFFER_SIZE ) {
        m_stores_outstanding=0;
        m_inst_in_pipeline=0;
        m_issued_inst = 0;
        reset();
    }
    static const unsigned IBUFFER_SIZE=2;
    class shader_core_ctx *m_shader;
    unsigned m_cta_id;
    unsigned m_warp_id;
    unsigned m_warp_size;
    unsigned m_dynamic_warp_id;

    address_type m_next_pc;
    unsigned n_completed;          // number of threads in warp completed
    std::bitset<MAX_WARP_SIZE> m_active_threads;

    bool m_imiss_pending;
    
    struct ibuffer_entry {
       ibuffer_entry() { m_valid = false; m_inst = NULL; }
       const warp_inst_t *m_inst;
       bool m_valid;
    };
    std::vector< ibuffer_entry* > m_ibuffer;
    unsigned m_next;
                                   
    unsigned m_n_atomic;           // number of outstanding atomic operations 
    bool     m_membar;             // if true, warp is waiting at memory barrier

    bool m_done_exit; // true once thread exit has been registered for threads in this warp

    unsigned long long m_last_fetch;

    unsigned m_stores_outstanding; // number of store requests sent but not yet acknowledged
    unsigned m_inst_in_pipeline;
    unsigned m_issued_inst;
};

class shd_warp_t : public base_shd_warp_t {
public:
    shd_warp_t( class shader_core_ctx *shader, unsigned warp_size )
        : base_shd_warp_t( shader, warp_size ) {
        for ( unsigned i = 0; i < IBUFFER_SIZE; ++i ) {
            base_shd_warp_t::m_ibuffer[ i ] = new ibuffer_entry();
        }
    }

    virtual ~shd_warp_t() {
        for ( unsigned i = 0; i < IBUFFER_SIZE; ++i ) {
            delete base_shd_warp_t::m_ibuffer[ i ];
        }
    }
};

class cache_conscious_ldst_unit;
class mem_allow_bit_shd_warp_t : public base_shd_warp_t {
public:
    mem_allow_bit_shd_warp_t( class shader_core_ctx *shader,
            unsigned warp_size,
            const cache_conscious_ldst_unit* ldst_unit );

    virtual ~mem_allow_bit_shd_warp_t() {
        for ( unsigned i = 0; i < IBUFFER_SIZE; ++i ) {
            delete m_ibuffer[ i ];
        }
    }

    virtual void ibuffer_fill( unsigned slot, const warp_inst_t *pI );

    virtual void ibuffer_flush() {
        for( unsigned i = 0; i < IBUFFER_SIZE; i++ ) {
            m_ibuffer[m_next]->m_mem_allow = true;
        }
        base_shd_warp_t::ibuffer_flush();
    }

    virtual void ibuffer_free() {
        m_ibuffer[m_next]->m_mem_allow = true;
        base_shd_warp_t::ibuffer_free();
    }

    void set_mem_allow() { m_ibuffer[m_next]->m_mem_allow = true; }
    bool check_mem_allow() { return m_ibuffer[m_next]->m_mem_allow; }

private:
    struct ibuffer_mem_allow_entry : public ibuffer_entry {
        ibuffer_mem_allow_entry() : m_mem_allow( true ){}
        bool m_mem_allow;
    };
    std::vector< ibuffer_mem_allow_entry* > m_ibuffer;
    const cache_conscious_ldst_unit* m_ldst_unit;
};

inline unsigned hw_tid_from_wid(unsigned wid, unsigned warp_size, unsigned i){return wid * warp_size + i;};
inline unsigned wid_from_hw_tid(unsigned tid, unsigned warp_size){return tid/warp_size;};

const unsigned WARP_PER_CTA_MAX = 48;

typedef std::bitset<WARP_PER_CTA_MAX> warp_set_t;

int register_bank(int regnum, int wid, unsigned num_banks, unsigned bank_warp_shift);

class shader_core_ctx;
class shader_core_config;
class shader_core_stats;

class scheduler_unit { //this can be copied freely, so can be used in std containers.
public:
    scheduler_unit(shader_core_stats* stats, shader_core_ctx* shader, 
                   Scoreboard* scoreboard, simt_stack** simt, 
                   std::vector<base_shd_warp_t*>* warp,
                   register_set* sp_out,
                   register_set* sfu_out,
                   register_set* mem_out) 
        : m_supervised_warps(), m_last_warp_issued( 0 ), m_stats(stats), m_shader(shader),
        m_scoreboard(scoreboard), m_simt_stack(simt), /*m_pipeline_reg(pipe_regs),*/ m_warp(warp),
        m_sp_out(sp_out),m_sfu_out(sfu_out),m_mem_out(mem_out), m_num_issued_last_cycle( 0 ) {}

    void add_supervised_warp_id( int i );

    virtual void cycle();

    void print_scheduled_list( FILE* file ) const {
        fprintf( file, "scheduler_unit::print_scheduled_list\n" );
        for ( std::vector<int>::const_iterator it = m_supervised_warps.begin();
                it != m_supervised_warps.end();
                ++it ) {
            fprintf( file, "%d, ", *it );
        }
        fprintf( file, "\n" );
    }

    void rr_order_warps();
    void greedy_top_order_warps();
    void greedy_oldest_order_warps();
    void greedy_until_stall_order_warps();

protected:
    base_shd_warp_t& warp(int i);

    std::vector<int> m_supervised_warps;
    int m_last_warp_issued;
    shader_core_stats *m_stats;
    shader_core_ctx* m_shader;
    // these things should become accessors: but would need a bigger rearchitect of how shader_core_ctx interacts with its parts.
    Scoreboard* m_scoreboard; 
    simt_stack** m_simt_stack;
    //warp_inst_t** m_pipeline_reg;
    std::vector<base_shd_warp_t*>* m_warp;
    register_set* m_sp_out;
    register_set* m_sfu_out;
    register_set* m_mem_out;
    unsigned m_num_issued_last_cycle;

private:
    virtual void order_warps() = 0;
    virtual void post_issue_attempt_process( unsigned num_issued ){}
    virtual bool should_stall_mem_op( const warp_inst_t* pI, unsigned warp_id ) { return false; }
};

enum warp_scheduler_t {
    LOOSE_ROUND_ROBIN_WARP_SCHEDULER = 0,
    GREEDY_TOP_WARP_SCHEDULER,                  //1
    CACHE_CONSCIOUS_WARP_SCHEDULER,             //2
    CUSTOM_MEMCACHED_WARP_SCHEDULER,            //3
    GREEDY_UNTIL_STALL_WARP_SCHEDULER,          //4
    NVIDIA_ISCA_2011_TWO_LEVEL_WARP_SCHEDULER,  //5
    GREEDY_THEN_OLDEST_WARP_SCHEDULER,          //6
    CONCURRENCY_LIMITED_SCHEDULER,              //7
    TEXAS_TECH_REPORT_SCHEDULER,                //8
    REAL_CACHE_CONSCIOUS_WARP_SCHEDULER,        //9
    REAL_MAB_CACHE_CONSCIOUS_WARP_SCHEDULER,    //10
    STRICT_ROUND_ROBIN_WARP_SCHEDULER,          //11
    POINT_SYSTEM_WARP_SCHEDULER                 //12
};

bool is_cc_scheduler( unsigned type );

enum warp_scheduling_policy {
    ROUND_ROBIN_WARP_SCHEDULER_POLICY = 0,
    GREEDY_TOP_WARP_SCHEDULER_POLICY,         //1
    GREEDY_THEN_OLDEST_WARP_SCHEDULER_POLICY, //2
    GREEDY_THEN_ROUND_ROBIN_SCHEDULER_POLICY  //3
};

class loose_round_robin_scheduler_unit : public scheduler_unit {
public:
    loose_round_robin_scheduler_unit(shader_core_stats* stats, shader_core_ctx* shader,
            Scoreboard* scoreboard, simt_stack** simt,
            std::vector<base_shd_warp_t*>* warp,
            register_set* sp_out,
            register_set* sfu_out,
            register_set* mem_out)
    : scheduler_unit( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out ) {}

    virtual ~loose_round_robin_scheduler_unit() {}

    virtual void order_warps();
};

class greepy_top_scheduler_unit : public scheduler_unit {
public:
    greepy_top_scheduler_unit(shader_core_stats* stats, shader_core_ctx* shader,
            Scoreboard* scoreboard, simt_stack** simt,
            std::vector<base_shd_warp_t*>* warp,
            register_set* sp_out,
            register_set* sfu_out,
            register_set* mem_out)
    : scheduler_unit( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out ) {}

    virtual ~greepy_top_scheduler_unit() {}

    virtual void order_warps();
};

struct cache_conscious_stats {
    cache_conscious_stats()
        : num_times_cache_probed( 0 ), num_times_throttler_accessed( 0 ) {}

    unsigned num_times_cache_probed;
    unsigned num_times_throttler_accessed;
};

class cache_conscious_ldst_unit;
class cache_conscious_scheduler_unit : public scheduler_unit {
public:
    cache_conscious_scheduler_unit(shader_core_stats* stats, shader_core_ctx* shader,
            Scoreboard* scoreboard, simt_stack** simt,
            std::vector<base_shd_warp_t*>* warp,
            register_set* sp_out,
            register_set* sfu_out,
            register_set* mem_out,
            cache_conscious_ldst_unit* dlst_unit,
            const thread_ctx_t* thread,
            unsigned workless_cycles_threshold );

    virtual ~cache_conscious_scheduler_unit() {}

    virtual void order_warps();

    virtual void post_issue_attempt_process( unsigned num_issued );

    bool is_stall_override_in_effect() const { return m_cycles_with_no_work_scheduled > m_workless_cycles_threshold; }

protected:
    const cache_conscious_ldst_unit* m_dlst_unit;
    const thread_ctx_t* m_thread;
    unsigned m_cycles_with_no_work_scheduled;
    unsigned m_workless_cycles_threshold;
};

class realistic_cache_conscious_scheduler_unit : public cache_conscious_scheduler_unit {
public:
    realistic_cache_conscious_scheduler_unit(shader_core_stats* stats, shader_core_ctx* shader,
                Scoreboard* scoreboard, simt_stack** simt,
                std::vector<base_shd_warp_t*>* warp,
                register_set* sp_out,
                register_set* sfu_out,
                register_set* mem_out,
                cache_conscious_ldst_unit* dlst_unit,
                const thread_ctx_t* thread,
                unsigned workless_cycles_threshold,
                unsigned num_cache_probes_per_cycle,
                warp_scheduling_policy policy )
        : cache_conscious_scheduler_unit( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out, dlst_unit, thread, workless_cycles_threshold ),
          m_max_cache_probes_per_cycle( num_cache_probes_per_cycle ), m_num_l1_probes_done_this_cycle( 0 ),
          m_warp_scheduling_policy( policy ) {}

    virtual void order_warps();

    virtual void cycle();

    virtual bool should_stall_mem_op( const warp_inst_t* pI, unsigned warp_id );

    void print_cc_stats( FILE* file ) const;

protected:
    unsigned num_cache_probes_per_cycle;
    cache_conscious_stats m_cc_stats;
    unsigned m_max_cache_probes_per_cycle;
    unsigned m_num_l1_probes_done_this_cycle;
    warp_scheduling_policy m_warp_scheduling_policy;
};

class realistic_memory_allow_bit_cc_scheduler_unit : public realistic_cache_conscious_scheduler_unit {
public:
    realistic_memory_allow_bit_cc_scheduler_unit(shader_core_stats* stats, shader_core_ctx* shader,
                Scoreboard* scoreboard, simt_stack** simt,
                std::vector<base_shd_warp_t*>* warp,
                register_set* sp_out,
                register_set* sfu_out,
                register_set* mem_out,
                cache_conscious_ldst_unit* dlst_unit,
                const thread_ctx_t* thread,
                unsigned workless_cycles_threshold,
                unsigned num_cache_probes_per_cycle,
                warp_scheduling_policy policy )
        : realistic_cache_conscious_scheduler_unit( stats, shader,
                scoreboard, simt, warp, sp_out, sfu_out, mem_out,
                dlst_unit, thread, workless_cycles_threshold,num_cache_probes_per_cycle,
                policy ), m_last_warp_processed( 0 ) {

    }

    virtual void cycle();

    virtual bool should_stall_mem_op( const warp_inst_t* pI, unsigned warp_id );

private:
    unsigned m_last_warp_processed;
};

class custom_memcached_scheduler_unit : public scheduler_unit {
public:
    custom_memcached_scheduler_unit(shader_core_stats* stats, shader_core_ctx* shader,
            Scoreboard* scoreboard, simt_stack** simt,
            std::vector<base_shd_warp_t*>* warp,
            register_set* sp_out,
            register_set* sfu_out,
            register_set* mem_out,
            unsigned l1_cache_size )
    : scheduler_unit( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out ),
      m_l1_cache_size( l1_cache_size ) {}

    virtual ~custom_memcached_scheduler_unit() {}

    virtual void order_warps();

private:
    unsigned m_l1_cache_size;
};

class greedy_until_stall_scheduler_unit : public scheduler_unit {
public:
    greedy_until_stall_scheduler_unit(shader_core_stats* stats, shader_core_ctx* shader,
            Scoreboard* scoreboard, simt_stack** simt,
            std::vector<base_shd_warp_t*>* warp,
            register_set* sp_out,
            register_set* sfu_out,
            register_set* mem_out)
    : scheduler_unit( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out ) {}

    virtual ~greedy_until_stall_scheduler_unit() {}

    virtual void order_warps();
};

class greedy_then_oldest_scheduler_unit : public scheduler_unit {
public:
    greedy_then_oldest_scheduler_unit(shader_core_stats* stats, shader_core_ctx* shader,
            Scoreboard* scoreboard, simt_stack** simt,
            std::vector<base_shd_warp_t*>* warp,
            register_set* sp_out,
            register_set* sfu_out,
            register_set* mem_out,
            unsigned total_num_warps )
    : scheduler_unit( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out ) {}

    virtual ~greedy_then_oldest_scheduler_unit() {}

    virtual void order_warps();
};

class nvidia_isca_2011_two_level_warp_scheduler : public scheduler_unit {
public:
    nvidia_isca_2011_two_level_warp_scheduler(shader_core_stats* stats, shader_core_ctx* shader,
            Scoreboard* scoreboard, simt_stack** simt,
            std::vector<base_shd_warp_t*>* warp,
            register_set* sp_out,
            register_set* sfu_out,
            register_set* mem_out,
            unsigned total_num_warps,
            unsigned num_active_warps,
            unsigned policy_type )
    : scheduler_unit( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out ),
      m_policy_type( warp_scheduling_policy( policy_type ) ),
      m_active_warps( num_active_warps ) {
        assert( num_active_warps > 0 );
        for ( unsigned i = 0; i < total_num_warps; ++i ) {
            if ( i < num_active_warps ) {
                m_active_warps_list.push_back( i );
            } else {
                m_pending_warps_list.push_back( i );
            }
        }
    }

    virtual ~nvidia_isca_2011_two_level_warp_scheduler() {}

    virtual void order_warps();
    virtual void post_issue_attempt_process( unsigned num_issued );
    void print_active_warp_list( FILE* file )  const {
        fprintf( file, "nvidia_isca_2011_two_level_warp_scheduler::print_active_warp_list\n" );
        for ( std::list< unsigned >::const_iterator it = m_active_warps_list.begin();
                it != m_active_warps_list.end(); ++it ) {
            fprintf( file, "%u, ", *it );
        }
        fprintf( file, "\n" );
    }

    void print_pending_warp_list( FILE* file ) const {
        fprintf( file, "nvidia_isca_2011_two_level_warp_scheduler::print_pending_warp_list\n" );
        for ( std::list< unsigned >::const_iterator it = m_pending_warps_list.begin();
                it != m_pending_warps_list.end(); ++it ) {
            fprintf( file, "%u, ", *it );
        }
        fprintf( file, "\n" );
    }

private:
    std::list< unsigned > m_active_warps_list;
    std::list< unsigned > m_pending_warps_list;
    warp_scheduling_policy m_policy_type;
    unsigned m_active_warps;
};

class concurrency_limited_scheduler_unit : public scheduler_unit {
public:
    concurrency_limited_scheduler_unit( shader_core_stats* stats, shader_core_ctx* shader,
            Scoreboard* scoreboard, simt_stack** simt,
            std::vector<base_shd_warp_t*>* warp,
            register_set* sp_out,
            register_set* sfu_out,
            register_set* mem_out,
            unsigned num_concurrent_warps,
            unsigned num_concurrent_ctas,
            unsigned warp_size );

    virtual ~concurrency_limited_scheduler_unit() {}

    virtual void order_warps();

private:
    unsigned m_max_concurrent_warps;
    unsigned m_max_concurrent_ctas;
    unsigned m_warp_size;
};

class texas_tech_report_scheduler_unit : public scheduler_unit {
public:
    texas_tech_report_scheduler_unit( shader_core_stats* stats, shader_core_ctx* shader,
                Scoreboard* scoreboard, simt_stack** simt,
                std::vector<base_shd_warp_t*>* warp,
                register_set* sp_out,
                register_set* sfu_out,
                register_set* mem_out,
                unsigned fetch_group_size );

    virtual ~texas_tech_report_scheduler_unit() {}

    virtual void order_warps();
private:
    unsigned m_fetch_group_size;
};

class strict_round_robin_scheduler_unit : public scheduler_unit {
public:
    strict_round_robin_scheduler_unit(shader_core_stats* stats, shader_core_ctx* shader,
            Scoreboard* scoreboard, simt_stack** simt,
            std::vector<base_shd_warp_t*>* warp,
            register_set* sp_out,
            register_set* sfu_out,
            register_set* mem_out)
    : scheduler_unit( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out ), m_current_turn_warp( 0 ) {}

    virtual ~strict_round_robin_scheduler_unit() {}

    virtual void order_warps();

private:
    unsigned m_current_turn_warp;
};

class point_system_scheduler_unit : public scheduler_unit {
public:
        point_system_scheduler_unit(
            shader_core_stats* stats, 
            shader_core_ctx* shader,
            Scoreboard* scoreboard, simt_stack** simt,
            std::vector<base_shd_warp_t*>* warp,
            register_set* sp_out,
            register_set* sfu_out,
            register_set* mem_out,
            scheduling_point_system* point_system)
    : scheduler_unit( stats, shader, scoreboard, simt, warp, sp_out, sfu_out, mem_out ), m_point_system(point_system) {}

    virtual void order_warps();

private:
    scheduling_point_system* m_point_system;
};

class opndcoll_rfu_t { // operand collector based register file unit
public:
   // constructors
   opndcoll_rfu_t()
   {
      m_num_banks=0;
      m_shader=NULL;
      m_initialized=false;
   }
   void add_cu_set(unsigned cu_set, unsigned num_cu, unsigned num_dispatch);
   typedef std::vector<register_set*> port_vector_t;
   typedef std::vector<unsigned int> uint_vector_t;
   void add_port( port_vector_t & input, port_vector_t & ouput, uint_vector_t cu_sets);
   void init( unsigned num_banks, shader_core_ctx *shader );

   // modifiers
   bool writeback( const warp_inst_t &warp ); // might cause stall 

   void step()
   {
        dispatch_ready_cu();   
        allocate_reads();
        for( unsigned p = 0 ; p < m_in_ports.size(); p++ ) 
            allocate_cu( p );
        process_banks();
   }

   void dump( FILE *fp ) const
   {
      fprintf(fp,"\n");
      fprintf(fp,"Operand Collector State:\n");
      for( unsigned n=0; n < m_cu.size(); n++ ) {
         fprintf(fp,"   CU-%2u: ", n);
         m_cu[n]->dump(fp,m_shader);
      }
      m_arbiter.dump(fp);
   }

   shader_core_ctx *shader_core() { return m_shader; }

private:

   void process_banks()
   {
      m_arbiter.reset_alloction();
   }

   void dispatch_ready_cu();
   void allocate_cu( unsigned port );
   void allocate_reads();

   // types

   class collector_unit_t;

   class op_t {
   public:

      op_t() { m_valid = false; }
      op_t( collector_unit_t *cu, unsigned op, unsigned reg, unsigned num_banks, unsigned bank_warp_shift )
      {
         m_valid = true;
         m_warp=NULL;
         m_cu = cu;
         m_operand = op;
         m_register = reg;
         m_bank = register_bank(reg,cu->get_warp_id(),num_banks,bank_warp_shift);
      }
      op_t( const warp_inst_t *warp, unsigned reg, unsigned num_banks, unsigned bank_warp_shift )
      {
         m_valid=true;
         m_warp=warp;
         m_register=reg;
         m_cu=NULL;
         m_operand = -1;
         m_bank = register_bank(reg,warp->warp_id(),num_banks,bank_warp_shift);
      }

      // accessors
      bool valid() const { return m_valid; }
      unsigned get_reg() const
      {
         assert( m_valid );
         return m_register;
      }
      unsigned get_wid() const
      {
          if( m_warp ) return m_warp->warp_id();
          else if( m_cu ) return m_cu->get_warp_id();
          else abort();
          return 0;
      }
      unsigned get_oc_id() const { return m_cu->get_id(); }
      unsigned get_bank() const { return m_bank; }
      unsigned get_operand() const { return m_operand; }
      void dump(FILE *fp) const 
      {
         if(m_cu) 
            fprintf(fp," <R%u, CU:%u, w:%02u> ", m_register,m_cu->get_id(),m_cu->get_warp_id());
         else if( !m_warp->empty() )
            fprintf(fp," <R%u, wid:%02u> ", m_register,m_warp->warp_id() );
      }
      std::string get_reg_string() const
      {
         char buffer[64];
         snprintf(buffer,64,"R%u", m_register);
         return std::string(buffer);
      }

      // modifiers
      void reset() { m_valid = false; }
   private:
      bool m_valid;
      collector_unit_t  *m_cu; 
      const warp_inst_t *m_warp;
      unsigned  m_operand; // operand offset in instruction. e.g., add r1,r2,r3; r2 is oprd 0, r3 is 1 (r1 is dst)
      unsigned  m_register;
      unsigned  m_bank;
   };

   enum alloc_t {
      NO_ALLOC,
      READ_ALLOC,
      WRITE_ALLOC,
   };

   class allocation_t {
   public:
      allocation_t() { m_allocation = NO_ALLOC; }
      bool is_read() const { return m_allocation==READ_ALLOC; }
      bool is_write() const {return m_allocation==WRITE_ALLOC; }
      bool is_free() const {return m_allocation==NO_ALLOC; }
      void dump(FILE *fp) const {
         if( m_allocation == NO_ALLOC ) { fprintf(fp,"<free>"); }
         else if( m_allocation == READ_ALLOC ) { fprintf(fp,"rd: "); m_op.dump(fp); }
         else if( m_allocation == WRITE_ALLOC ) { fprintf(fp,"wr: "); m_op.dump(fp); }
         fprintf(fp,"\n");
      }
      void alloc_read( const op_t &op )  { assert(is_free()); m_allocation=READ_ALLOC; m_op=op; }
      void alloc_write( const op_t &op ) { assert(is_free()); m_allocation=WRITE_ALLOC; m_op=op; }
      void reset() { m_allocation = NO_ALLOC; }
   private:
      enum alloc_t m_allocation;
      op_t m_op;
   };

   class arbiter_t {
   public:
      // constructors
      arbiter_t()
      {
         m_queue=NULL;
         m_allocated_bank=NULL;
         m_allocator_rr_head=NULL;
         _inmatch=NULL;
         _outmatch=NULL;
         _request=NULL;
         m_last_cu=0;
      }
      void init( unsigned num_cu, unsigned num_banks ) 
      { 
         assert(num_cu > 0);
         assert(num_banks > 0);
         m_num_collectors = num_cu;
         m_num_banks = num_banks;
         _inmatch = new int[ m_num_banks ];
         _outmatch = new int[ m_num_collectors ];
         _request = new int*[ m_num_banks ];
         for(unsigned i=0; i<m_num_banks;i++) 
             _request[i] = new int[m_num_collectors];
         m_queue = new std::list<op_t>[num_banks];
         m_allocated_bank = new allocation_t[num_banks];
         m_allocator_rr_head = new unsigned[num_cu];
         for( unsigned n=0; n<num_cu;n++ ) 
            m_allocator_rr_head[n] = n%num_banks;
         reset_alloction();
      }

      // accessors
      void dump(FILE *fp) const
      {
         fprintf(fp,"\n");
         fprintf(fp,"  Arbiter State:\n");
         fprintf(fp,"  requests:\n");
         for( unsigned b=0; b<m_num_banks; b++ ) {
            fprintf(fp,"    bank %u : ", b );
            std::list<op_t>::const_iterator o = m_queue[b].begin();
            for(; o != m_queue[b].end(); o++ ) {
               o->dump(fp);
            }
            fprintf(fp,"\n");
         }
         fprintf(fp,"  grants:\n");
         for(unsigned b=0;b<m_num_banks;b++) {
            fprintf(fp,"    bank %u : ", b );
            m_allocated_bank[b].dump(fp);
         }
         fprintf(fp,"\n");
      }

      // modifiers
      std::list<op_t> allocate_reads(); 

      void add_read_requests( collector_unit_t *cu ) 
      {
         const op_t *src = cu->get_operands();
         for( unsigned i=0; i<MAX_REG_OPERANDS*2; i++) {
            const op_t &op = src[i];
            if( op.valid() ) {
               unsigned bank = op.get_bank();
               m_queue[bank].push_back(op);
            }
         }
      }
      bool bank_idle( unsigned bank ) const
      {
          return m_allocated_bank[bank].is_free();
      }
      void allocate_bank_for_write( unsigned bank, const op_t &op )
      {
         assert( bank < m_num_banks );
         m_allocated_bank[bank].alloc_write(op);
      }
      void allocate_for_read( unsigned bank, const op_t &op )
      {
         assert( bank < m_num_banks );
         m_allocated_bank[bank].alloc_read(op);
      }
      void reset_alloction()
      {
         for( unsigned b=0; b < m_num_banks; b++ ) 
            m_allocated_bank[b].reset();
      }

   private:
      unsigned m_num_banks;
      unsigned m_num_collectors;

      allocation_t *m_allocated_bank; // bank # -> register that wins
      std::list<op_t> *m_queue;

      unsigned *m_allocator_rr_head; // cu # -> next bank to check for request (rr-arb)
      unsigned  m_last_cu; // first cu to check while arb-ing banks (rr)

      int *_inmatch;
      int *_outmatch;
      int **_request;
   };

   class input_port_t {
   public:
       input_port_t(port_vector_t & input, port_vector_t & output, uint_vector_t cu_sets)
       : m_in(input),m_out(output), m_cu_sets(cu_sets)
       {
           assert(input.size() == output.size());
           assert(not m_cu_sets.empty());
       }
   //private:
       port_vector_t m_in,m_out;
       uint_vector_t m_cu_sets;
   };

   class collector_unit_t {
   public:
      // constructors
      collector_unit_t()
      { 
         m_free = true;
         m_warp = NULL;
         m_output_register = NULL;
         m_src_op = new op_t[MAX_REG_OPERANDS*2];
         m_not_ready.reset();
         m_warp_id = -1;
         m_num_banks = 0;
         m_bank_warp_shift = 0;
      }
      // accessors
      bool ready() const;
      const op_t *get_operands() const { return m_src_op; }
      void dump(FILE *fp, const shader_core_ctx *shader ) const;

      unsigned get_warp_id() const { return m_warp_id; }
      unsigned get_id() const { return m_cuid; } // returns CU hw id

      // modifiers
      void init(unsigned n, 
                unsigned num_banks, 
                unsigned log2_warp_size,
                const core_config *config,
                opndcoll_rfu_t *rfu ); 
      bool allocate( register_set* pipeline_reg, register_set* output_reg );

      void collect_operand( unsigned op )
      {
         m_not_ready.reset(op);
      }

      void dispatch();
      bool is_free(){return m_free;}

   private:
      bool m_free;
      unsigned m_cuid; // collector unit hw id
      unsigned m_warp_id;
      warp_inst_t  *m_warp;
      register_set* m_output_register; // pipeline register to issue to when ready
      op_t *m_src_op;
      std::bitset<MAX_REG_OPERANDS*2> m_not_ready;
      unsigned m_num_banks;
      unsigned m_bank_warp_shift;
      opndcoll_rfu_t *m_rfu;

   };

   class dispatch_unit_t {
   public:
      dispatch_unit_t(std::vector<collector_unit_t>* cus) 
      { 
         m_last_cu=0;
         m_collector_units=cus;
         m_num_collectors = (*cus).size();
         m_next_cu=0;
      }

      collector_unit_t *find_ready()
      {
         for( unsigned n=0; n < m_num_collectors; n++ ) {
            unsigned c=(m_last_cu+n+1)%m_num_collectors;
            if( (*m_collector_units)[c].ready() ) {
               m_last_cu=c;
               return &((*m_collector_units)[c]);
            }
         }
         return NULL;
      }

   private:
      unsigned m_num_collectors;
      std::vector<collector_unit_t>* m_collector_units;
      unsigned m_last_cu; // dispatch ready cu's rr
      unsigned m_next_cu;  // for initialization
   };

   // opndcoll_rfu_t data members
   bool m_initialized;

   unsigned m_num_collector_sets;
   //unsigned m_num_collectors;
   unsigned m_num_banks;
   unsigned m_bank_warp_shift;
   unsigned m_warp_size;
   std::vector<collector_unit_t *> m_cu;
   arbiter_t m_arbiter;

   //unsigned m_num_ports;
   //std::vector<warp_inst_t**> m_input;
   //std::vector<warp_inst_t**> m_output;
   //std::vector<unsigned> m_num_collector_units;
   //warp_inst_t **m_alu_port;

   std::vector<input_port_t> m_in_ports;
   typedef std::map<unsigned /* collector set */, std::vector<collector_unit_t> /*collector sets*/ > cu_sets_t;
   cu_sets_t m_cus;
   std::vector<dispatch_unit_t> m_dispatch_units;

   //typedef std::map<warp_inst_t**/*port*/,dispatch_unit_t> port_to_du_t;
   //port_to_du_t                     m_dispatch_units;
   //std::map<warp_inst_t**,std::list<collector_unit_t*> > m_free_cu;
   shader_core_ctx                 *m_shader;
};

class barrier_set_t {
public:
   barrier_set_t( unsigned max_warps_per_core, unsigned max_cta_per_core );

   // during cta allocation
   void allocate_barrier( unsigned cta_id, warp_set_t warps );

   // during cta deallocation
   void deallocate_barrier( unsigned cta_id );

   typedef std::map<unsigned, warp_set_t >  cta_to_warp_t;

   // individual warp hits barrier
   void warp_reaches_barrier( unsigned cta_id, unsigned warp_id );

   // fetching a warp
   bool available_for_fetch( unsigned warp_id ) const;

   // warp reaches exit 
   void warp_exit( unsigned warp_id );

   // assertions
   bool warp_waiting_at_barrier( unsigned warp_id ) const;

   // debug
   void dump() const;

private:
   unsigned m_max_cta_per_core;
   unsigned m_max_warps_per_core;

   cta_to_warp_t m_cta_to_warps; 
   warp_set_t m_warp_active;
   warp_set_t m_warp_at_barrier;
};

struct insn_latency_info {
   unsigned pc;
   unsigned long latency;
};

struct ifetch_buffer_t {
    ifetch_buffer_t() { m_valid=false; }

    ifetch_buffer_t( address_type pc, unsigned nbytes, unsigned warp_id ) 
    { 
        m_valid=true; 
        m_pc=pc; 
        m_nbytes=nbytes; 
        m_warp_id=warp_id;
    }

    bool m_valid;
    address_type m_pc;
    unsigned m_nbytes;
    unsigned m_warp_id;
};

class shader_core_config;

class simd_function_unit {
public:
    simd_function_unit( const shader_core_config *config );
    ~simd_function_unit() { delete m_dispatch_reg; }

    // modifiers
    virtual void issue( register_set& source_reg ) { source_reg.move_out_to(m_dispatch_reg); occupied.set(m_dispatch_reg->latency); m_new_dispatch_reg = true;}
    virtual void cycle() = 0;

    // accessors
    virtual unsigned clock_multiplier() const { return 1; }
    virtual bool can_issue( const warp_inst_t &inst ) const { return m_dispatch_reg->empty() && !occupied.test(inst.latency); }
    virtual bool stallable() const = 0;
    virtual void print( FILE *fp ) const
    {
        fprintf(fp,"%s dispatch= ", m_name.c_str() );
        m_dispatch_reg->print(fp);
    }
protected:
    std::string m_name;
    const shader_core_config *m_config;
    warp_inst_t *m_dispatch_reg;
    static const unsigned MAX_ALU_LATENCY = 512;
    std::bitset<MAX_ALU_LATENCY> occupied;
    bool m_new_dispatch_reg;
};

class pipelined_simd_unit : public simd_function_unit {
public:
    pipelined_simd_unit( register_set* result_port, const shader_core_config *config, unsigned max_latency );

    //modifiers
    virtual void cycle() 
    {
        if( !m_pipeline_reg[0]->empty() ){
            m_result_port->move_in(m_pipeline_reg[0]);
        }
        for( unsigned stage=0; (stage+1)<m_pipeline_depth; stage++ ) 
            move_warp(m_pipeline_reg[stage], m_pipeline_reg[stage+1]);
        if( !m_dispatch_reg->empty() ) {
            if( !m_dispatch_reg->dispatch_delay()) {
                int start_stage = m_dispatch_reg->latency - m_dispatch_reg->initiation_interval;
                move_warp(m_pipeline_reg[start_stage],m_dispatch_reg);
            }
        }
        occupied >>=1;
    }

    virtual void issue( register_set& source_reg )
    {
        //move_warp(m_dispatch_reg,source_reg);
        //source_reg.move_out_to(m_dispatch_reg);
        simd_function_unit::issue(source_reg);
        m_new_dispatch_reg = true;
    }

    // accessors
    virtual bool stallable() const { return false; }
    virtual bool can_issue( const warp_inst_t &inst ) const
    {
        return simd_function_unit::can_issue(inst);
    }
    virtual void print(FILE *fp) const
    {
        simd_function_unit::print(fp);
        for( int s=m_pipeline_depth-1; s>=0; s-- ) {
            if( !m_pipeline_reg[s]->empty() ) { 
                fprintf(fp,"      %s[%2d] ", m_name.c_str(), s );
                m_pipeline_reg[s]->print(fp);
            }
        }
    }
protected:
    unsigned m_pipeline_depth;
    warp_inst_t **m_pipeline_reg;
    register_set *m_result_port;
};

class sfu : public pipelined_simd_unit
{
public:
    sfu( register_set* result_port, const shader_core_config *config );
    virtual bool can_issue( const warp_inst_t &inst ) const
    {
        switch(inst.op) {
        case SFU_OP: break;
        case ALU_SFU_OP: break;
        default: return false;
        }
        return pipelined_simd_unit::can_issue(inst);
    }
};

class sp_unit : public pipelined_simd_unit
{
public:
    sp_unit( register_set* result_port, const shader_core_config *config );
    virtual bool can_issue( const warp_inst_t &inst ) const
    {
        switch(inst.op) {
        case SFU_OP: return false; 
        case LOAD_OP: return false;
        case STORE_OP: return false;
        case MEMORY_BARRIER_OP: return false;
        default: break;
        }
        return pipelined_simd_unit::can_issue(inst);
    }
};

class simt_core_cluster;
class shader_memory_interface;
class shader_core_mem_fetch_allocator;
class cache_t;

/*
enum mem_div_status {
   MEM_DIV_INVALID,
   MEM_DIV_NEVERUSED,
   MEM_DIV_USEDATARRIVAL,
   MEM_DIV_USEDLATER
};
*/
typedef struct _mem_div_info{
	unsigned useful_bytes[8];
        unsigned already_useful_bytes[8];

	unsigned useful_later_bytes;
	unsigned already_useful_later_bytes;

	unsigned useful_later_bytes_read_now;

	unsigned total_bytes;
	unsigned total_unique_bytes;

	unsigned never_used_bytes;


	unsigned num_mem_access;
	unsigned cycle_start;
	unsigned cycle_end;

}mem_div_info;

class shader_tlb_interface : public read_only_cache {
public:
   shader_tlb_interface( const char *name,
         const tlb_config &config,
         int core_id, int type_id,
         mem_fetch_interface *memport,
         enum mem_fetch_status status,
         gpgpu_sim* the_gpu );
   virtual ~shader_tlb_interface(){}

   virtual void cycle() = 0;
   virtual cache_request_status translate_addrs( warp_inst_t& instr );
   virtual cache_request_status access( new_addr_type virtual_mem_addr, new_addr_type& linear_mem_addr, virtual_page_type& page_translation ) = 0;
   virtual void print( FILE* fp ) const;

protected:
   std::map< unsigned, unsigned > m_objects_accessed;
   std::map< new_addr_type, unsigned > m_cache_lines_accessed;
   const tlb_config &m_config;
   gpgpu_sim* m_the_gpu;
};

// TLB that does nothing and always hits
class shader_ideal_tlb : public shader_tlb_interface {
public:
   shader_ideal_tlb( const char *name,
         const tlb_config &config,
         int core_id, int type_id,
         mem_fetch_interface *memport,
         enum mem_fetch_status status,
         gpgpu_sim* the_gpu );
   virtual ~shader_ideal_tlb(){}

   virtual void cycle(){}
   virtual cache_request_status access( new_addr_type virtual_mem_addr, new_addr_type& linear_mem_addr, virtual_page_type& page_translation );
};

class reorder_tlb_stats {
public:
    reorder_tlb_stats() : m_num_removed_page_mappings( 0 ) {}

    void remove_page_event() { ++m_num_removed_page_mappings; }

    void print( FILE* fp ) const {
        fprintf( fp, "reorder_tlb_stats::m_num_removed_page_mappings = %u\n", m_num_removed_page_mappings );
    }
private:
    unsigned m_num_removed_page_mappings;
};

class shader_page_reordering_tlb : public shader_tlb_interface {
public:
   shader_page_reordering_tlb( const char *name,
         const tlb_config &config,
         int core_id, int type_id,
         mem_fetch_interface *memport,
         enum mem_fetch_status status,
         gpgpu_sim* the_gpu,
         const vm_policy_manager* policy_man,
         virtual_policy_cache* dl1 );
   virtual ~shader_page_reordering_tlb(){}

   virtual void cycle();
   //virtual cache_request_status translate_addrs( warp_inst_t& instr );
   virtual cache_request_status access( new_addr_type virtual_mem_addr, new_addr_type& linear_mem_addr, virtual_page_type& page_translation );

   cache_request_status access( new_addr_type virtual_mem_addr, base_virtual_page*& policy_used ) const;

   // Function meant to be used by the vm_manager that pushes in a new region and policy
   void insert_new_policy( const vm_page_mapping_payload& policy );

   virtual void print( FILE* fp ) const {
       shader_tlb_interface::print( fp );
       m_stats.print( fp );
   }

private:
   enum state {
      NOT_FLUSHING_L1,
      FLUSHING_L1,
   };
   struct policy_flush_entry {
       vm_page_mapping_payload mapping;
      new_addr_type next_line_to_flushed;
   };

   void flush_invalid_L1_lines( const vm_page_mapping_payload& policy );

private:
   const vm_policy_manager* m_policy_manager;
   state m_current_state;
   std::list< policy_flush_entry > m_flush_entries;
   virtual_policy_cache* m_dl1_cache;
   reorder_tlb_stats m_stats;
};

class ldst_unit: public pipelined_simd_unit {
public:
    ldst_unit( mem_fetch_interface *icnt,
               shader_core_mem_fetch_allocator *mf_allocator,
               shader_core_ctx *core, 
               opndcoll_rfu_t *operand_collector,
               Scoreboard *scoreboard,
               const shader_core_config *config, 
               const memory_config *mem_config,  
               class shader_core_stats *stats, 
               unsigned sid, unsigned tpc );

    // modifiers
    virtual void issue( register_set &inst );
    virtual void cycle();
     
    void fill( mem_fetch *mf );
    void flush();
    void writeback();

    // accessors
    virtual unsigned clock_multiplier() const;

    virtual bool can_issue( const warp_inst_t &inst ) const
    {
        switch(inst.op) {
        case LOAD_OP: break;
        case STORE_OP: break;
        case MEMORY_BARRIER_OP: break;
        default: return false;
        }
        return m_dispatch_reg->empty();
    }
    virtual bool stallable() const { return true; }
    bool response_buffer_full() const;
    void print(FILE *fout) const;
    void print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses );


    void final_eval_cache_lines(){
       if ( m_L1D )
           m_L1D->get_stats().final_eval_cache_lines();
    }
    unsigned get_never_used(){
       if ( m_L1D )
           return m_L1D->get_stats().get_never_used();
       else
           return 0;
    }
    unsigned get_used_now(){
       if ( m_L1D )
           return m_L1D->get_stats().get_used_now();
       else
           return 0;
    }
    unsigned get_event_used(){
       if ( m_L1D )
           return m_L1D->get_stats().get_event_used();
       else
           return 0;
    }
  
    unsigned get_used_cache_array( unsigned idx ){
       if ( m_L1D )
           return m_L1D->get_stats().get_used_cache_array(idx);
       else
           return 0;
    }
    unsigned get_event_used_cache_array( unsigned idx ){
       if ( m_L1D )
           return m_L1D->get_stats().get_event_used_cache_array(idx);
       else
           return 0;
    }


protected:
   bool shared_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   bool constant_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   bool texture_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);
   virtual bool memory_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);

   mem_stage_stall_type process_memory_access_queue( cache_t *cache, warp_inst_t &inst );

   void collect_mem_divergence_statistics( const warp_inst_t &inst, const mem_access_t &access );
   void mem_div_new_access ( const mem_access_t &access, addr_t access_pc, unsigned long warp_id );
   mem_div_status mem_div_read_byte(const new_addr_type address, const bool using_now);
   void mem_div_access_byte( new_addr_type address, bool using_now );
   void use_shared_cache_system( data_cache* dcache, warp_inst_t &inst, cache_request_status& status, mem_fetch *mf, std::list<cache_event> &events );

   const memory_config *m_memory_config;
   class mem_fetch_interface *m_icnt;
   shader_core_mem_fetch_allocator *m_mf_allocator;
   class shader_core_ctx *m_core;
   unsigned m_sid;
   unsigned m_tpc;

   tex_cache *m_L1T; // texture cache
   read_only_cache *m_L1C; // constant cache
   evict_on_write_cache *m_L1D; // data cache
   evict_on_write_cache **m_L1D_per_warp; // per warp data cache
   tag_array *m_shared_data_only_L1D; // data cache
   shader_tlb_interface* m_tlb;
   std::map<unsigned/*warp_id*/, std::map<unsigned/*regnum*/,unsigned/*count*/> > m_pending_writes;
   std::list<mem_fetch*> m_response_fifo;
   opndcoll_rfu_t *m_operand_collector;
   Scoreboard *m_scoreboard;

   mem_fetch *m_next_global;
   warp_inst_t m_next_wb;
   unsigned m_writeback_arb; // round-robin arbiter for writeback contention between L1T, L1C, shared
   unsigned m_num_writeback_clients;

   enum mem_stage_stall_type m_mem_rc;

   shader_core_stats *m_stats;

   // for debugging
   unsigned long long m_last_inst_gpu_sim_cycle;
   unsigned long long m_last_inst_gpu_tot_sim_cycle;

   std::map <new_addr_type, div_map_t> mem_div_map_ldst;
   vm_policy_manager *m_vm_manager;
   std::list< std::pair<mem_fetch*,int> > m_hit_queue;

};

class simplified_mem_system;

class simplified_mem_system_ldst_unit : public ldst_unit {
public:
    simplified_mem_system_ldst_unit( mem_fetch_interface *icnt,
                                     shader_core_mem_fetch_allocator *mf_allocator,
                                     shader_core_ctx *core,
                                     opndcoll_rfu_t *operand_collector,
                                     Scoreboard *scoreboard,
                                     const shader_core_config *config,
                                     const memory_config *mem_config,
                                     shader_core_stats *stats,
                                     unsigned sid,
                                     unsigned tpc,
                                     simplified_mem_system* simple_mem_sys )
        : ldst_unit( icnt, mf_allocator, core, operand_collector, scoreboard, config, mem_config, stats, sid, tpc ),
          m_simplified_mem_system( simple_mem_sys ) {}

    virtual void cycle();
protected:
    virtual bool memory_cycle( warp_inst_t &inst, mem_stage_stall_type &rc_fail, mem_stage_access_type &fail_type);

    simplified_mem_system* m_simplified_mem_system;
};

class cache_conscious_ldst_unit : public ldst_unit {
public:
    cache_conscious_ldst_unit( mem_fetch_interface *icnt,
                   shader_core_mem_fetch_allocator *mf_allocator,
                   shader_core_ctx *core,
                   opndcoll_rfu_t *operand_collector,
                   Scoreboard *scoreboard,
                   const shader_core_config *config,
                   const memory_config *mem_config,
                   class shader_core_stats *stats,
                   unsigned sid, unsigned tpc )
        : ldst_unit( icnt, mf_allocator, core, operand_collector, scoreboard, config, mem_config, stats, sid, tpc ){}

    cache_request_status probe_l1( new_addr_type addr ) const {
        return m_L1D->probe( addr );
    }
    bool should_whitelist_warp( unsigned warp_id ) const {
        return m_scheduler->is_stall_override_in_effect()
                && dynamic_cast< high_locality_protected_cache* >( m_L1D )->should_whitelist_warp( warp_id );
    }

    void set_scheduler( const cache_conscious_scheduler_unit* scheduler ) { m_scheduler = scheduler; }

private:
    const cache_conscious_scheduler_unit* m_scheduler;
};

class point_system_ldst_unit : public ldst_unit {
public:
    point_system_ldst_unit( mem_fetch_interface *icnt,
                   shader_core_mem_fetch_allocator *mf_allocator,
                   shader_core_ctx *core,
                   opndcoll_rfu_t *operand_collector,
                   Scoreboard *scoreboard,
                   const shader_core_config *config,
                   const memory_config *mem_config,
                   class shader_core_stats *stats,
                   unsigned sid, unsigned tpc, 
                   scheduling_point_system* point_system )
        : ldst_unit( icnt, mf_allocator, core, operand_collector, scoreboard, config, mem_config, stats, sid, tpc )
        {
            high_locality_protected_cache* cache = dynamic_cast<high_locality_protected_cache*>(m_L1D);
            if (cache) {
                cache->set_point_system(point_system);
            } else {
                fprintf(stderr, "point_system_ldst_unit expects a high_locality_protected_cache");
                abort();
            }
        }

};


enum pipeline_stage_name_t {
    ID_OC_SP=0,
    ID_OC_SFU,  
    ID_OC_MEM,  
    OC_EX_SP,
    OC_EX_SFU,
    OC_EX_MEM,
    EX_WB,
    N_PIPELINE_STAGES 
};

const char* const pipeline_stage_name_decode[] = {
    "ID_OC_SP",
    "ID_OC_SFU",  
    "ID_OC_MEM",  
    "OC_EX_SP",
    "OC_EX_SFU",
    "OC_EX_MEM",
    "EX_WB",
    "N_PIPELINE_STAGES" 
};

struct warp_scheduler_config : public core_config {
    warp_scheduler_config()
        : m_scheduler_type( 0 ), m_scheduling_policy( 0 ), m_active_warps( 0 ),
          m_workless_cycles_threshold( 0xFFFFFFFF ), m_max_l1_probes_per_cycle( 0xFFFFFFFF ),
          m_active_ctas( 0 ) {}

    void init() {
        int ntok = sscanf(m_config_string,"%u:%u:%u:%u:%u:%u",
                        &m_scheduler_type,
                        &m_scheduling_policy,
                        &m_active_warps,
                        &m_workless_cycles_threshold,
                        &m_max_l1_probes_per_cycle,
                        &m_active_ctas);
        if(ntok < 2) {
            printf("GPGPU-Sim uArch: error while parsing configuration string warp_scheduler_config\n");
            abort();
        }
    }

    unsigned m_scheduler_type;
    unsigned m_scheduling_policy;
    unsigned m_active_warps;
    unsigned m_workless_cycles_threshold;
    unsigned m_max_l1_probes_per_cycle;
    unsigned m_active_ctas;
    char* m_config_string;
};

struct simplified_mem_system_config {
    void init() {
        if ( strncmp( config_string, "off", 3 ) == 0 ) {
            enabled = false;
        } else {
            enabled = true;
            unsigned ret = sscanf( config_string,"%d,%d", &latency_to_mem, &maximum_num_outstanding_requests );
            assert( ret == 2 );
        }
    }

    bool enabled;
    unsigned latency_to_mem;
    unsigned maximum_num_outstanding_requests;

    char* config_string;
};

struct shader_core_config : public core_config
{
    shader_core_config(){
	pipeline_widths_string = NULL;
    }
    void init()
    {
        int ntok = sscanf(gpgpu_shader_core_pipeline_opt,"%d:%d",
                          &n_thread_per_shader,
                          &warp_size);
        if(ntok != 2) {
           printf("GPGPU-Sim uArch: error while parsing configuration string gpgpu_shader_core_pipeline_opt\n");
           abort();
	}

	char* toks = new char[100];
	char* tokd = toks;
	strcpy(toks,pipeline_widths_string);

	toks = strtok(toks,",");
	for (unsigned i = 0; i < N_PIPELINE_STAGES; i++) { 
	    assert(toks);
	    ntok = sscanf(toks,"%d", &pipe_widths[i]);
	    assert(ntok == 1); 
	    toks = strtok(NULL,",");
	}
	delete tokd;

        if (n_thread_per_shader > MAX_THREAD_PER_SM) {
           printf("GPGPU-Sim uArch: Error ** increase MAX_THREAD_PER_SM in abstract_hardware_model.h from %u to %u\n", 
                  MAX_THREAD_PER_SM, n_thread_per_shader);
           abort();
        }
        max_warps_per_shader =  n_thread_per_shader/warp_size;
        assert( !(n_thread_per_shader % warp_size) );
        max_sfu_latency = 512;
        max_sp_latency = 32;
        m_L1I_config.init();
        m_L1T_config.init();
        m_L1C_config.init();
        m_L1D_config.init();
        m_L1D_shared_only_config.init();
        m_L1D_per_warp_individual_cache.init();
        m_high_locality_L1D_config.init();
        m_L1DVC_config.init();
        m_TLB_config.init();
        m_vm_manager_config.init();
        m_warp_scheduler_config.init();
        m_simplified_config.init();
        m_point_system_config.init();
        gpgpu_cache_texl1_linesize = m_L1T_config.get_line_sz();
        gpgpu_cache_constl1_linesize = m_L1C_config.get_line_sz();
        m_valid = true;
        mem_div_cycles_threshold = 10000;
    }
    void reg_options(class OptionParser * opp );
    unsigned max_cta( const kernel_info_t &k ) const;
    unsigned num_shader() const { return n_simt_clusters*n_simt_cores_per_cluster; }
    unsigned sid_to_cluster( unsigned sid ) const { return sid / n_simt_cores_per_cluster; }
    unsigned sid_to_cid( unsigned sid )     const { return sid % n_simt_cores_per_cluster; }
    unsigned cid_to_sid( unsigned cid, unsigned cluster_id ) const { return cluster_id*n_simt_cores_per_cluster + cid; }

// data
    char *gpgpu_shader_core_pipeline_opt;
    bool gpgpu_perfect_mem;
    enum divergence_support_t model;
    unsigned n_thread_per_shader;
    unsigned max_warps_per_shader; 
    unsigned max_cta_per_core; //Limit on number of concurrent CTAs in shader core

    char* pipeline_widths_string;
    int pipe_widths[N_PIPELINE_STAGES];

    cache_config m_L1I_config;
    cache_config m_L1T_config;
    cache_config m_L1C_config;
    cache_config m_L1D_config;
    cache_config m_L1D_shared_only_config;
    cache_config m_L1D_per_warp_individual_cache;
    high_locality_cache_config m_high_locality_L1D_config;
    cache_config m_L1DVC_config;
    tlb_config m_TLB_config;
    vm_manager_config m_vm_manager_config;
    warp_scheduler_config m_warp_scheduler_config;
    scheduling_point_system_config m_point_system_config;
    
    bool gpgpu_dwf_reg_bankconflict;

    int gpgpu_num_sched_per_core;
    int gpgpu_max_insn_issue_per_warp;

    //op collector
    int gpgpu_operand_collector_num_units_sp;
    int gpgpu_operand_collector_num_units_sfu;
    int gpgpu_operand_collector_num_units_mem;
    int gpgpu_operand_collector_num_units_gen;

    unsigned int gpgpu_operand_collector_num_in_ports_sp;
    unsigned int gpgpu_operand_collector_num_in_ports_sfu;
    unsigned int gpgpu_operand_collector_num_in_ports_mem;
    unsigned int gpgpu_operand_collector_num_in_ports_gen;

    unsigned int gpgpu_operand_collector_num_out_ports_sp;
    unsigned int gpgpu_operand_collector_num_out_ports_sfu;
    unsigned int gpgpu_operand_collector_num_out_ports_mem;
    unsigned int gpgpu_operand_collector_num_out_ports_gen;

    int gpgpu_num_sp_units;
    int gpgpu_num_sfu_units;
    int gpgpu_num_mem_units;	

    //Shader core resources
    unsigned gpgpu_shader_registers;
    int gpgpu_warpdistro_shader;
    unsigned gpgpu_num_reg_banks;
    bool gpgpu_reg_bank_use_warp_id;
    bool gpgpu_local_mem_map;
    
    unsigned max_sp_latency;
    unsigned max_sfu_latency;
    
    unsigned n_simt_cores_per_cluster;
    unsigned n_simt_clusters;
    unsigned n_simt_ejection_buffer_size;
    unsigned ldst_unit_response_queue_size;
    int simt_core_sim_order; 
    unsigned mem_div_cycles_threshold;
    bool gpgpu_struct_vm_reorg;
    unsigned gpgpu_warp_issue_shader;
    unsigned gpgpu_warp_cache_distro_shader;
    unsigned gpgpu_shared_only_sharers_for_shared;    
    shader_warp_ifetch_order gpgpu_instr_fetch_order;
    simplified_mem_system_config m_simplified_config;

    bool gpgpu_mem_visualizer_enabled;

    unsigned mem2device(unsigned memid) const { return memid + n_simt_clusters; }
};

struct block_access_stats {
    std::map< unsigned, unsigned > warps_hitting_line; // warp to number of L1 hits
    std::map< unsigned, unsigned > pcs_hitting_line; // pc to number of L1 hits
    std::map< unsigned, unsigned > pc_exclusive_hits; // only count this if the pc loading the line and the pc hitting the line are the same
    std::map< unsigned, std::list< unsigned > > warp_to_hit_times;
    std::map< unsigned, std::list< unsigned > > pc_to_hit_times;
    std::vector< unsigned > threads_hitting_line;
};

struct shader_core_stats_pod {
	void* shader_core_stats_pod_start[]; // DO NOT MOVE FROM THE TOP - spaceless pointer to the start of this structure
    unsigned *m_num_sim_insn; // number of scalar thread instructions committed by this shader core
    unsigned *m_num_sim_winsn; // number of warp instructions committed by this shader core
    unsigned *m_n_diverge;    // number of divergence occurring in this shader
    unsigned gpgpu_n_load_insn;
    unsigned gpgpu_n_store_insn;
    unsigned gpgpu_n_shmem_insn;
    unsigned gpgpu_n_tex_insn;
    unsigned gpgpu_n_const_insn;
    unsigned gpgpu_n_param_insn;
    unsigned gpgpu_n_shmem_bkconflict;
    unsigned gpgpu_n_cache_bkconflict;
    int      gpgpu_n_intrawarp_mshr_merge;
    unsigned gpgpu_n_cmem_portconflict;
    unsigned gpu_stall_shd_mem_breakdown[N_MEM_STAGE_ACCESS_TYPE][N_MEM_STAGE_STALL_TYPE];
    unsigned gpu_reg_bank_conflict_stalls;
    unsigned *shader_cycle_distro;
    unsigned *last_shader_cycle_distro;
    unsigned *num_warps_issuable;
    unsigned gpgpu_n_stall_shd_mem;

    //memory access classification
    int gpgpu_n_mem_read_local;
    int gpgpu_n_mem_write_local;
    int gpgpu_n_mem_texture;
    int gpgpu_n_mem_const;
    int gpgpu_n_mem_read_global;
    int gpgpu_n_mem_write_global;
    int gpgpu_n_mem_read_inst;
    
    unsigned made_write_mfs;
    unsigned made_read_mfs;

    // stats to keep track of mem_divergence 
    unsigned long long old_tot_useful_bytes_read;
    unsigned long long old_tot_bytes_read;
    
    // divergence stats
    unsigned long long tot_bytes_read;
    unsigned long long tot_useful_bytes_read;
    unsigned long long tot_eventually_useful_bytes_read;
    unsigned long long tot_already_eventually_useful_bytes_read;
    unsigned long long tot_already_useful_bytes_read;
    
    unsigned bytes_used_now;
    unsigned already_bytes_used_now;
    unsigned total_bytes_read;
    unsigned total_unique_bytes_read;

    // Same thing as above, but no history
    unsigned t_bytes_read;
    unsigned t_bytes_used[8];
    unsigned t_mem_access;

    // divergence cache stats
    unsigned long long mem_div_hit_bytes;
    unsigned long long mem_div_access_num_bytes_requested;
    double mem_div_accesses;
    double mem_div_divergent_accesses;

    // This is used to determine what is the maximum thread number assigned to this core
    // i.e if this core runs at most 2 32-wide warps, then this number will be 64
    unsigned m_max_thread_assigned;
};

struct mem_div_load_counters {
	unsigned long long num_accesses;
	unsigned long long num_times_all_bytes_used_or_eventually_used;
	unsigned long long num_times_gte_3_quarters_bytes_used_or_eventually_used;
	unsigned long long num_times_gte_half_bytes_used_or_eventually_used;
	unsigned long long num_times_gte_1_quarter_used_or_eventually_used;
};


class shader_core_stats : private shader_core_stats_pod {
public:
    shader_core_stats( const shader_core_config *config, const print_streams* print_streamer, dynamic_per_pc_stats* dynamic_per_pc_stats )
        : m_print_streams( print_streamer ), m_dynamic_per_pc_stats( dynamic_per_pc_stats )
    {
        m_config = config;
        shader_core_stats_pod *pod = reinterpret_cast< shader_core_stats_pod * > ( this->shader_core_stats_pod_start );
        memset(pod,0,sizeof(shader_core_stats_pod));

        m_num_sim_insn = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_num_sim_winsn = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        m_n_diverge = (unsigned*) calloc(config->num_shader(),sizeof(unsigned));
        shader_cycle_distro = (unsigned*) calloc(config->warp_size+3, sizeof(unsigned));
        last_shader_cycle_distro = (unsigned*) calloc(m_config->warp_size+3, sizeof(unsigned));

        m_recently_issued_warps.resize( config->num_shader() );
        m_block_address_stats.resize( config->num_shader() );
        m_num_shared_only_cache_hits.resize( config->num_shader(), 0 );
        m_num_shared_only_cache_insertions.resize( config->num_shader(), 0 );

        m_shader_dynamic_warp_issue_distro.resize( config->num_shader() );
        m_shader_warp_issue_distro.resize( config->num_shader() );
        m_last_shader_warp_issue_distro.resize( config->num_shader() );
        for ( unsigned i = 0; i < config->num_shader(); ++i ) {
            m_shader_warp_issue_distro[ i ] = ( unsigned* )calloc( config->max_warps_per_shader + 1, sizeof( unsigned ) );
            m_last_shader_warp_issue_distro[ i ] = ( unsigned* )calloc( config->max_warps_per_shader + 1, sizeof( unsigned ) );
        }

        m_shader_vc_hit_distro.resize( config->num_shader() );
        m_last_shader_vc_hit_distro.resize( config->num_shader() );
        for ( unsigned i = 0; i < config->num_shader(); ++i ) {
            m_shader_vc_hit_distro[ i ] = ( unsigned* )calloc( config->max_warps_per_shader + 1, sizeof( unsigned ) );
            m_last_shader_vc_hit_distro[ i ] = ( unsigned* )calloc( config->max_warps_per_shader + 1, sizeof( unsigned ) );
        }

        m_shader_dl1_s0_per_warp_distro.resize( config->max_warps_per_shader );
        m_last_shader_dl1_s0_per_warp_distro.resize( config->max_warps_per_shader );
        for ( unsigned i = 0; i < config->max_warps_per_shader; ++i ) {
            m_shader_dl1_s0_per_warp_distro[ i ] = ( unsigned* )calloc( NUM_CACHE_REQUEST_STATUS, sizeof( unsigned ) );
            m_last_shader_dl1_s0_per_warp_distro[ i ] = ( unsigned* )calloc( NUM_CACHE_REQUEST_STATUS, sizeof( unsigned ) );
        }


        m_shader_l1_data_cache_distro.resize( config->num_shader() );
        m_last_shader_l1_data_cache_distro.resize( config->num_shader() );
        for ( unsigned i = 0; i < config->num_shader(); ++i ) {
            m_shader_l1_data_cache_distro[ i ] = ( unsigned* )calloc( NUM_CACHE_REQUEST_STATUS, sizeof( unsigned ) );
            m_last_shader_l1_data_cache_distro[ i ] = ( unsigned* )calloc( NUM_CACHE_REQUEST_STATUS, sizeof( unsigned ) );
        }
    }

    virtual ~shader_core_stats() {
        free( m_num_sim_insn );
        free( m_n_diverge );
        free( shader_cycle_distro );
        free( last_shader_cycle_distro );

        for ( unsigned i = 0; i < m_config->num_shader(); ++i ) {
            free( m_shader_warp_issue_distro[ i ] );
            free( m_last_shader_warp_issue_distro[ i ] );
        }
    }

    void new_grid()
    {
    }

    void visualizer_print( gzFile visualizer_file );

    void print( FILE *fout ) const;

    void print_recent_warp_issued( FILE *fout, unsigned sid ) const;

    void event_warp_issued( unsigned s_id, unsigned warp_id, unsigned num_issued, unsigned dynamic_warp_id ) {
        m_recently_issued_warps[ s_id ].push_back( warp_id );

        if ( m_recently_issued_warps[ s_id ].size() > RECENTLY_ISSUED_WARPS_TO_RECORD ) {
            m_recently_issued_warps[ s_id ].pop_front();
        }

        assert( warp_id <= m_config->max_warps_per_shader );
        for ( unsigned i = 0; i < num_issued; ++i ) {
            ++m_shader_warp_issue_distro[ s_id ][ warp_id + 1 ];
            if ( m_shader_dynamic_warp_issue_distro[ s_id ].size() <= dynamic_warp_id ) {
                m_shader_dynamic_warp_issue_distro[ s_id ].resize(dynamic_warp_id + 1);
            }
            ++m_shader_dynamic_warp_issue_distro[ s_id ][ dynamic_warp_id ];
        }
    }

    void record_cache_block_hit( new_addr_type block_addr, unsigned sid, unsigned warp_id, unsigned pc, const active_mask_t& warp_mask );

    void print_useful_byte_information() const;

    dynamic_per_pc_stats* get_dynamic_per_pc_stats() { return m_dynamic_per_pc_stats; }

    void print_memory_visualization() const;

private:
    static const unsigned RECENTLY_ISSUED_WARPS_TO_RECORD = 50;

    const shader_core_config *m_config;
    std::map <new_addr_type/*byte address*/,div_map_t/*used before?*/ > mem_div_map;
    std::map < int, std::map< new_addr_type, mem_div_load_counters > > warp_to_pc_to_load_counters_map;
    std::map< new_addr_type, mem_div_load_counters > all_warps_pc_to_load_counters_map;
    const print_streams* m_print_streams;

    // This is the same as m_shader_warp issue_distro, but instead
    // of counting the instrs issued for each static warp it counts
    // the instructions issued for each dynamic warp.
    std::vector< std::vector<unsigned> > m_shader_dynamic_warp_issue_distro;

    std::vector< unsigned* > m_shader_warp_issue_distro;
    std::vector< unsigned* > m_last_shader_warp_issue_distro;

    std::vector< unsigned* > m_shader_vc_hit_distro;
    std::vector< unsigned* > m_last_shader_vc_hit_distro;

    std::vector< unsigned* > m_shader_dl1_s0_per_warp_distro;
    std::vector< unsigned* > m_last_shader_dl1_s0_per_warp_distro;

    std::vector< unsigned* > m_shader_l1_data_cache_distro;
    std::vector< unsigned* > m_last_shader_l1_data_cache_distro;

    // warp issue history
    std::vector< std::list< unsigned > > m_recently_issued_warps;

    // vector (per shader core) of maps that map the block address to a collection of stats about that particular address
    std::vector< std::map< new_addr_type /* block address */, block_access_stats > > m_block_address_stats;
    std::vector< unsigned > m_num_shared_only_cache_hits;
    std::vector< unsigned > m_num_shared_only_cache_insertions;

    std::vector<mem_div_info> m_div_info;

    dynamic_per_pc_stats* m_dynamic_per_pc_stats;

    friend class shader_core_ctx;
    friend class ldst_unit;
    friend class simt_core_cluster;
    friend class scheduler_unit;
    friend class loose_round_robin_scheduler_unit;
    friend class cache_conscious_scheduler_unit;
    friend class greepy_top_scheduler_unit;
    friend class custom_memcached_scheduler_unit;
    friend class mem_div_cache;
    friend class simplified_mem_system_ldst_unit;
    friend class scheduling_point_system;
};

class shader_core_mem_fetch_allocator : public mem_fetch_allocator {
public:
    shader_core_mem_fetch_allocator( unsigned core_id, unsigned cluster_id, const memory_config *config )
    {
    	m_core_id = core_id;
    	m_cluster_id = cluster_id;
    	m_memory_config = config;
    }
    mem_fetch *alloc( new_addr_type addr, mem_access_type type, unsigned size, bool wr, mem_fetch_payload *payload ) const 
    {
    	mem_access_t access( type, addr, size, wr );
    	mem_fetch *mf = new mem_fetch( access, 
    				       NULL,
    				       wr?WRITE_PACKET_SIZE:READ_PACKET_SIZE, 
    				       -1, 
    				       m_core_id, 
    				       m_cluster_id,
    				       m_memory_config,
                           payload );
    	return mf;
    }
    
    mem_fetch *alloc( const warp_inst_t &inst, const mem_access_t &access, mem_fetch_payload *payload = NULL ) const
    {
        warp_inst_t inst_copy = inst;
        mem_fetch *mf = new mem_fetch(access, 
                                      &inst_copy, 
                                      access.is_write()?WRITE_PACKET_SIZE:READ_PACKET_SIZE,
                                      inst.warp_id(),
                                      m_core_id, 
                                      m_cluster_id, 
                                      m_memory_config,
                                      payload);
        return mf;
    }

private:
    unsigned m_core_id;
    unsigned m_cluster_id;
    const memory_config *m_memory_config;
};

class shader_core_ctx : public core_t {
public:
    // creator:
    shader_core_ctx( class gpgpu_sim *gpu,
                     class simt_core_cluster *cluster,
                     unsigned shader_id,
                     unsigned tpc_id,
                     const struct shader_core_config *config,
                     const struct memory_config *mem_config,
                     shader_core_stats *stats );

// used by simt_core_cluster:
    // modifiers
    void cycle();
    void reinit(unsigned start_thread, unsigned end_thread, bool reset_not_completed );
    void issue_block2core( class kernel_info_t &kernel );
    void cache_flush();
    void accept_fetch_response( mem_fetch *mf );
    void accept_ldst_unit_response( class mem_fetch * mf );
    void set_kernel( kernel_info_t *k ) 
    {
        assert(k);
        m_kernel=k; 
        k->inc_running(); 
        printf("GPGPU-Sim uArch: Shader %d bind to kernel %u \'%s\'\n", m_sid, m_kernel->get_uid(),
                 m_kernel->name().c_str() );
    }
    
    // accessors
    bool fetch_unit_response_buffer_full() const;
    bool ldst_unit_response_buffer_full() const;
    unsigned get_not_completed() const { return m_not_completed; }
    unsigned get_n_active_cta() const { return m_n_active_cta; }
    kernel_info_t *get_kernel() { return m_kernel; }

// used by functional simulation:
    // modifiers
    virtual void warp_exit( unsigned warp_id );
    
    // accessors
    virtual bool warp_waiting_at_barrier( unsigned warp_id ) const;
    void get_pdom_stack_top_info( unsigned tid, unsigned *pc, unsigned *rpc ) const;

// used by pipeline timing model components:
    // modifiers
    void mem_instruction_stats(const warp_inst_t &inst);
    void decrement_atomic_count( unsigned wid, unsigned n );
    void inc_store_req( unsigned warp_id) { m_warp[warp_id]->inc_store_req(); }
    void dec_inst_in_pipeline( unsigned warp_id ) { m_warp[warp_id]->dec_inst_in_pipeline(); } // also used in writeback()
    void store_ack( class mem_fetch *mf );
    bool warp_waiting_at_mem_barrier( unsigned warp_id );
    void set_max_cta( const kernel_info_t &kernel );
    void warp_inst_complete(const warp_inst_t &inst); 
    
    // accessors
    std::list<unsigned> get_regs_written( const inst_t &fvt ) const;
    const shader_core_config *get_config() const { return m_config; }
    void print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses );

    void issue_warp( warp_inst_t *&warp, const warp_inst_t *pI, const active_mask_t &active_mask, unsigned warp_id );

    unsigned get_sid() { return m_sid; }

// debug:
    void display_simt_state(FILE *fout, int mask ) const;
    void display_pipeline( FILE *fout, int print_mem, int mask3bit ) const;

//    void print_data_cache(){m_ldst_unit->print_L1D();}
    void final_eval_cache_lines(){
	m_ldst_unit->final_eval_cache_lines();
    }
    unsigned get_never_used(){
	return m_ldst_unit->get_never_used();
    }
    unsigned get_used_now(){
	return m_ldst_unit->get_used_now();
    }
    unsigned get_event_used(){
	return m_ldst_unit->get_event_used();
    }
    unsigned get_used_cache_array(unsigned idx){
	return m_ldst_unit->get_used_cache_array(idx);
    }
    unsigned get_event_used_cache_array(unsigned idx){
	return m_ldst_unit->get_event_used_cache_array(idx);
    }

    ptx_thread_info* get_thread_state(int thread_id)
    { return m_thread[thread_id]; }

private:
    int test_res_bus(int latency);
    void init_warps(unsigned cta_id, unsigned start_thread, unsigned end_thread);
    virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t, unsigned tid);
    address_type next_pc( int tid ) const;
    std::list< unsigned > get_fetch_order() const;
    void fetch();
    void register_cta_thread_exit( unsigned cta_num );

    void decode();
    
    void issue();
    friend class scheduler_unit; //this is needed to use private issue warp.
    void issue_warp( register_set& warp, const warp_inst_t *pI, const active_mask_t &active_mask, unsigned warp_id );
    void func_exec_inst( warp_inst_t &inst );

     // Returns numbers of addresses in translated_addrs
    unsigned translate_local_memaddr( address_type localaddr, unsigned tid, unsigned num_shader, unsigned datasize, new_addr_type* translated_addrs );

    void read_operands();
    
    void execute();
    
    void writeback();
    
    // used in display_pipeline():
    void dump_warp_state( FILE *fout ) const;
    void print_stage(unsigned int stage, FILE *fout) const;
    unsigned long long m_last_inst_gpu_sim_cycle;
    unsigned long long m_last_inst_gpu_tot_sim_cycle;

    // general information
    unsigned m_sid; // shader id
    unsigned m_tpc; // texture processor cluster id (aka, node id when using interconnect concentration)
    const shader_core_config *m_config;
    const memory_config *m_memory_config;
    class simt_core_cluster *m_cluster;

    // statistics 
    shader_core_stats *m_stats;

    // CTA scheduling / hardware thread allocation
    unsigned m_n_active_cta; // number of Cooperative Thread Arrays (blocks) currently running on this shader.
    unsigned m_cta_status[MAX_CTA_PER_SHADER]; // CTAs status 
    unsigned m_not_completed; // number of threads to be completed (==0 when all thread on this core completed) 
    std::bitset<MAX_THREAD_PER_SM> m_active_threads;
    
    // thread contexts 
    thread_ctx_t             *m_threadState;
    
    // interconnect interface
    mem_fetch_interface *m_icnt;
    shader_core_mem_fetch_allocator *m_mem_fetch_allocator;
    
    // fetch
    read_only_cache *m_L1I; // instruction cache
    int  m_last_warp_fetched;

    // decode/dispatch
    std::vector<base_shd_warp_t*>   m_warp;   // per warp information array
    barrier_set_t             m_barriers;
    ifetch_buffer_t           m_inst_fetch_buffer;
    std::vector<register_set> m_pipeline_reg;
    Scoreboard               *m_scoreboard;
    opndcoll_rfu_t            m_operand_collector;

    //schedule
    std::vector<scheduler_unit*>  schedulers;

    // execute
    unsigned m_num_function_units;
    std::vector<pipeline_stage_name_t> m_dispatch_port;
    std::vector<pipeline_stage_name_t> m_issue_port;
    std::vector<simd_function_unit*> m_fu; // stallable pipelines should be last in this array
    ldst_unit *m_ldst_unit;
    static const unsigned MAX_ALU_LATENCY = 512;
    unsigned num_result_bus;
    std::vector< std::bitset<MAX_ALU_LATENCY>* > m_result_bus;

    // used for local address mapping with single kernel launch
    unsigned kernel_max_cta_per_shader;
    unsigned kernel_padded_threads_per_cta;
    // Used for handing out dynamic warp_ids to new warps.
    // the differnece between a warp_id and a dynamic_warp_id
    // is that the dynamic_warp_id is a running number unique to every warp
    // run on this shader, where the warp_id is the static warp slot.
    unsigned m_dynamic_warp_id; 
    scheduling_point_system m_scheduling_point_system;
};

class simt_core_cluster {
public:
    simt_core_cluster( class gpgpu_sim *gpu, 
                       unsigned cluster_id, 
                       const struct shader_core_config *config, 
                       const struct memory_config *mem_config,
                       shader_core_stats *stats,
                       memory_stats_t *mstats );

    void core_cycle();
    void icnt_cycle();

    void reinit();
    unsigned issue_block2core();
    void cache_flush();
    bool icnt_injection_buffer_full(unsigned size, bool write);
    void icnt_inject_request_packet(class mem_fetch *mf);

    // for perfect memory interface
    bool response_queue_full() {
        return ( m_response_fifo.size() >= m_config->n_simt_ejection_buffer_size );
    }
    void push_response_fifo(class mem_fetch *mf) {
        m_response_fifo.push_back(mf);
    }

    void get_pdom_stack_top_info( unsigned sid, unsigned tid, unsigned *pc, unsigned *rpc ) const;
    unsigned max_cta( const kernel_info_t &kernel );
    unsigned get_not_completed() const;
    void print_not_completed( FILE *fp ) const;
    unsigned get_n_active_cta() const;
    gpgpu_sim *get_gpu() { return m_gpu; }

    void display_pipeline( unsigned sid, FILE *fout, int print_mem, int mask );
    void print_cache_stats( FILE *fp, unsigned& dl1_accesses, unsigned& dl1_misses ) const;

    std::map< address_type, sourceline_and_hit_num >* get_pointer_load_info( int core, int thread_id );
    std::list< bin_with_redundant_count >* get_bin_redundant_count( int core, int thread_id );
    unsigned long long get_num_dynamic_load_info( int core, int thread_id );
    unsigned long long get_num_dynamic_pointer_load_info( int core, int thread_id );

    void final_eval_cache_lines();

    unsigned get_all_never_used();
    unsigned get_all_used_now();
    unsigned get_all_event_used();

    unsigned get_used_cache_array(unsigned idx);
    unsigned get_event_used_cache_array(unsigned idx);

private:
    unsigned m_cluster_id;
    gpgpu_sim *m_gpu;
    const shader_core_config *m_config;
    shader_core_stats *m_stats;
    memory_stats_t *m_memory_stats;
    shader_core_ctx **m_core;

    unsigned m_cta_issue_next_core;
    std::list<unsigned> m_core_sim_order;
    std::list<mem_fetch*> m_response_fifo;
};

class shader_memory_interface : public mem_fetch_interface {
public:
    shader_memory_interface( shader_core_ctx *core, simt_core_cluster *cluster ) { m_core=core; m_cluster=cluster; }
    virtual bool full( unsigned size, bool write, mem_fetch* fetch = NULL ) const
    {
        return m_cluster->icnt_injection_buffer_full(size,write);
    }
    virtual void push(mem_fetch *mf) 
    {
        m_cluster->icnt_inject_request_packet(mf);
    }
private:
    shader_core_ctx *m_core;
    simt_core_cluster *m_cluster;
};

class perfect_memory_interface : public mem_fetch_interface {
public:
    perfect_memory_interface( shader_core_ctx *core, simt_core_cluster *cluster ) { m_core=core; m_cluster=cluster; }
    virtual bool full( unsigned size, bool write, mem_fetch* fetch = NULL ) const
    {
        return m_cluster->response_queue_full();
    }
    virtual void push(mem_fetch *mf)
    {
        if( !mf->get_inst().empty() )
            m_core->mem_instruction_stats(mf->get_inst()); // not I$-fetch
        if ( mf && mf->isatomic() )
            mf->do_atomic(); // execute atomic inside the "memory subsystem"
        m_cluster->push_response_fifo(mf);
    }
private:
    shader_core_ctx *m_core;
    simt_core_cluster *m_cluster;
};

#endif /* SHADER_H */
