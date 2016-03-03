// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh, Timothy Rogers,
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

#include "abstract_hardware_model.h"
#include "cuda-sim/memory.h"
#include "option_parser.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx-stats.h"
#include "cuda-sim/cuda-sim.h"
#include "gpgpu-sim/gpu-sim.h"
#include <algorithm>
#include <math.h>
 
unsigned mem_access_t::sm_next_access_uid = 0;   
unsigned warp_inst_t::sm_next_uid = 0;

extern int g_use_host_memory_space;
translation_config g_translation_config = NO_TRANSLATION;

// Currently we only support one type of translated page.
virtual_page_type g_translated_page_type = VP_NON_TRANSLATED;

void move_warp( warp_inst_t *&dst, warp_inst_t *&src )
{
   assert( dst->empty() );
   warp_inst_t* temp = dst;
   dst = src;
   src = temp;
   src->clear();
}

void gpgpu_functional_sim_config::reg_options(class OptionParser * opp)
{
	option_parser_register(opp, "-gpgpu_ptx_use_cuobjdump", OPT_BOOL,
                 &m_ptx_use_cuobjdump,
                 "Use cuobjdump to extract ptx and sass from binaries",
#if (CUDART_VERSION >= 4000)
                 "1"
#else
                 "0"
#endif
                 );
    option_parser_register(opp, "-gpgpu_ptx_convert_to_ptxplus", OPT_BOOL,
                 &m_ptx_convert_to_ptxplus,
                 "Convert SASS (native ISA) to ptxplus and run ptxplus",
                 "0");
    option_parser_register(opp, "-gpgpu_ptx_force_max_capability", OPT_UINT32,
                 &m_ptx_force_max_capability,
                 "Force maximum compute capability",
                 "0");
   option_parser_register(opp, "-gpgpu_ptx_inst_debug_to_file", OPT_BOOL, 
                &g_ptx_inst_debug_to_file, 
                "Dump executed instructions' debug information to file", 
                "0");
   option_parser_register(opp, "-gpgpu_ptx_inst_debug_file", OPT_CSTR, &g_ptx_inst_debug_file, 
                  "Executed instructions' debug output file",
                  "inst_debug.txt");
   option_parser_register(opp, "-gpgpu_ptx_inst_debug_thread_uid", OPT_INT32, &g_ptx_inst_debug_thread_uid, 
               "Thread UID for executed instructions' debug output", 
               "1");
}

void gpgpu_functional_sim_config::ptx_set_tex_cache_linesize(unsigned linesize)
{
   m_texcache_linesize = linesize;
}

gpgpu_t::gpgpu_t( const gpgpu_functional_sim_config &config )
    : m_function_model_config(config)
{
   if (g_use_host_memory_space) {
	   m_global_mem = new memory_space_emu("global");
   } else {
	   m_global_mem = new memory_space_impl<8192>("global",64*1024);
   }
   m_tex_mem = new memory_space_impl<8192>("tex",64*1024);
   m_surf_mem = new memory_space_impl<8192>("surf",64*1024);

   m_dev_malloc=GLOBAL_HEAP_START; 

   if(m_function_model_config.get_ptx_inst_debug_to_file() != 0) 
      ptx_inst_debug_file = fopen(m_function_model_config.get_ptx_inst_debug_file(), "w");
}

address_type line_size_based_tag_func(new_addr_type address, new_addr_type line_size)
{
   //gives the tag for an address based on a given line size
   return address & ~(line_size-1);
}

void warp_inst_t::clear_active( const active_mask_t &inactive ) {
    active_mask_t test = m_warp_active_mask;
    test &= inactive;
    assert( test == inactive ); // verify threads being disabled were active
    m_warp_active_mask &= ~inactive;
}

void warp_inst_t::set_not_active( unsigned lane_id ) {
    m_warp_active_mask.reset(lane_id);
}

void warp_inst_t::set_active( const active_mask_t &active ) {
   m_warp_active_mask = active;
   if( m_isatomic ) {
      for( unsigned i=0; i < m_config->warp_size; i++ ) {
         if( !m_warp_active_mask.test(i) ) {
             m_per_scalar_thread[i].callback.function = NULL;
             m_per_scalar_thread[i].callback.instruction = NULL;
             m_per_scalar_thread[i].callback.thread = NULL;
         }
      }
   }
}

void warp_inst_t::do_atomic(bool forceDo) {
    do_atomic( m_warp_active_mask,forceDo );
}

void warp_inst_t::do_atomic( const active_mask_t& access_mask,bool forceDo ) {
    assert( m_isatomic && (!m_empty||forceDo) );
    for( unsigned i=0; i < m_config->warp_size; i++ )
    {
        if( access_mask.test(i) )
        {
            dram_callback_t &cb = m_per_scalar_thread[i].callback;
            if( cb.thread )
                cb.function(cb.instruction, cb.thread);
        }
    }
}

warp_inst_t::per_thread_info::per_thread_info() {
    for(unsigned i=0; i<8; i++)
        memreqaddr[i] = 0;
}

// HACK STATS
unsigned long long g_hack_num_xactions_gen = 0;
unsigned long long g_hack_num_warp_accesses_processed = 0;

void warp_inst_t::generate_mem_accesses()
{
    if( empty() || op == MEMORY_BARRIER_OP || m_mem_accesses_created ) 
        return;
    if ( !((op == LOAD_OP) || (op == STORE_OP)) )
        return; 
    if( m_warp_active_mask.count() == 0 ) 
        return; // predicated off

    const size_t starting_queue_size = m_accessq.size();

    assert( is_load() || is_store() );
    assert( m_per_scalar_thread_valid ); // need address information per thread

    bool is_write = is_store();

    mem_access_type access_type;
    switch (space.get_type()) {
    case const_space:
    case param_space_kernel: 
        access_type = CONST_ACC_R; 
        break;
    case tex_space: 
        access_type = TEXTURE_ACC_R;   
        break;
    case global_space:       
        access_type = is_write? GLOBAL_ACC_W: GLOBAL_ACC_R;   
        break;
    case local_space:
    case param_space_local:  
        access_type = is_write? LOCAL_ACC_W: LOCAL_ACC_R;   
        break;
    case shared_space: break;
    default: assert(0); break; 
    }

    // Calculate memory accesses generated by this warp
    new_addr_type cache_block_size = 0; // in bytes 
	
    switch( space.get_type() ) {
    case shared_space: {
        unsigned subwarp_size = m_config->warp_size / m_config->mem_warp_parts;
        unsigned total_accesses=0;
        for( unsigned subwarp=0; subwarp <  m_config->mem_warp_parts; subwarp++ ) {

            // data structures used per part warp 
            std::map<unsigned,std::map<new_addr_type,unsigned> > bank_accs; // bank -> word address -> access count

            // step 1: compute accesses to words in banks
            for( unsigned thread=subwarp*subwarp_size; thread < (subwarp+1)*subwarp_size; thread++ ) {
                if( !active(thread) ) 
                    continue;
                new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
                //FIXME: deferred allocation of shared memory should not accumulate across kernel launches
                //assert( addr < m_config->gpgpu_shmem_size ); 
                unsigned bank = m_config->shmem_bank_func(addr);
                new_addr_type word = line_size_based_tag_func(addr,m_config->WORD_SIZE);
                bank_accs[bank][word]++;
            }

            // step 2: look for and select a broadcast bank/word if one occurs
            bool broadcast_detected = false;
            new_addr_type broadcast_word=(new_addr_type)-1;
            unsigned broadcast_bank=(unsigned)-1;
            std::map<unsigned,std::map<new_addr_type,unsigned> >::iterator b;
            for( b=bank_accs.begin(); b != bank_accs.end(); b++ ) {
                unsigned bank = b->first;
                std::map<new_addr_type,unsigned> &access_set = b->second;
                std::map<new_addr_type,unsigned>::iterator w;
                for( w=access_set.begin(); w != access_set.end(); ++w ) {
                    if( w->second > 1 ) {
                        // found a broadcast
                        broadcast_detected=true;
                        broadcast_bank=bank;
                        broadcast_word=w->first;
                        break;
                    }
                }
                if( broadcast_detected ) 
                    break;
            }

            // step 3: figure out max bank accesses performed, taking account of broadcast case
            unsigned max_bank_accesses=0;
            for( b=bank_accs.begin(); b != bank_accs.end(); b++ ) {
                unsigned bank_accesses=0;
                std::map<new_addr_type,unsigned> &access_set = b->second;
                std::map<new_addr_type,unsigned>::iterator w;
                for( w=access_set.begin(); w != access_set.end(); ++w ) 
                    bank_accesses += w->second;
                if( broadcast_detected && broadcast_bank == b->first ) {
                    for( w=access_set.begin(); w != access_set.end(); ++w ) {
                        if( w->first == broadcast_word ) {
                            unsigned n = w->second;
                            assert(n > 1); // or this wasn't a broadcast
                            assert(bank_accesses >= (n-1));
                            bank_accesses -= (n-1);
                            break;
                        }
                    }
                }
                if( bank_accesses > max_bank_accesses ) 
                    max_bank_accesses = bank_accesses;
            }

            // step 4: accumulate
            total_accesses+= max_bank_accesses;
        }
        assert( total_accesses > 0 && total_accesses <= m_config->warp_size );
        cycles = total_accesses; // shared memory conflicts modeled as larger initiation interval 
        ptx_file_line_stats_add_smem_bank_conflict( pc, total_accesses );
        break;
    }

    case tex_space: 
        cache_block_size = m_config->gpgpu_cache_texl1_linesize;
        break;
    case const_space:  case param_space_kernel:
        cache_block_size = m_config->gpgpu_cache_constl1_linesize; 
        break;

    case global_space: case local_space: case param_space_local:
        if( m_config->gpgpu_coalesce_arch == 13 ) {
           if(isatomic())
               memory_coalescing_arch_13_atomic(is_write, access_type);
           else
               memory_coalescing_arch_13(is_write, access_type);
        } else abort();

        break;

    default:
        abort();
    }

    if( cache_block_size ) {
        assert( m_accessq.empty() );
        mem_access_byte_mask_t byte_mask; 
        std::map<new_addr_type,active_mask_t> accesses; // block address -> set of thread offsets in warp
        std::map<new_addr_type,active_mask_t>::iterator a;
        for( unsigned thread=0; thread < m_config->warp_size; thread++ ) {
            if( !active(thread) ) 
                continue;
            new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
            unsigned block_address = line_size_based_tag_func(addr,cache_block_size);
            accesses[block_address].set(thread);
            unsigned idx = addr-block_address; 
            for( unsigned i=0; i < data_size; i++ ) 
                byte_mask.set(idx+i);
        }
        for( a=accesses.begin(); a != accesses.end(); ++a ) 
            m_accessq.push_back( mem_access_t(access_type,a->first,cache_block_size,is_write,a->second,byte_mask) );
    }

    if ( space.get_type() == global_space ) {
        ptx_file_line_stats_add_uncoalesced_gmem( pc, m_accessq.size() - starting_queue_size );
    }
    m_mem_accesses_created=true;
}

std::map< unsigned, unsigned > g_objects_accessed;
std::map< new_addr_type, unsigned > g_cache_lines_accessed;

#define ENABLE_TRANSLATION_PRINTS 0

new_addr_type virtual_unaligned_hot_page::translate( new_addr_type virtual_addr ) const {
   // All of this will be relative to alignment
   new_addr_type result;
   const size_t page_size = (new_addr_type)m_end - (new_addr_type)m_start;
   const new_addr_type A = m_start_of_first_page;
   const new_addr_type b = virtual_addr;
   const new_addr_type b_page = b / page_size * page_size;
   size_t alignment;
   if ( m_start_of_first_page == b_page ) {
      alignment = 0;
   } else {
      alignment = A
            + ceil( (float)page_size / (float)m_object_size * floor( (float)( b - A ) / (float)page_size ) ) * m_object_size
            - b_page;
   }

   if ( virtual_addr - (new_addr_type)m_start < alignment ) { return virtual_addr; }
   else if ( (new_addr_type)m_end - virtual_addr < m_object_size - alignment ) { return virtual_addr; }
   else if ( m_hotbytes.count() + FORMAT_DATA_SIZE > m_cache_block_size ) { return virtual_addr; }
   else {
      const size_t k = ( virtual_addr - ( alignment + (new_addr_type)m_start ) ) / m_object_size;
      const size_t b = ( virtual_addr - ( alignment + (new_addr_type)m_start ) ) % m_object_size;
      const size_t c = m_cache_block_size / m_hotbytes.count();
      unsigned bytes_in = 0;
      unsigned hot_bytes_in = 0;
      for ( unsigned i = 0; i < m_hotbytes.size(); ++i ) {
         if ( m_hotbytes.test( i ) ) {
            if ( bytes_in == b ) break;
            ++hot_bytes_in;
            // you are not hot, bail out
         } else if ( bytes_in == b ) {
             return virtual_addr;
         }
         ++bytes_in;
      }
      result = (new_addr_type)m_start + ( k / c ) * m_cache_block_size + ( k % c ) * m_hotbytes.count() + hot_bytes_in;
   }
#if ENABLE_TRANSLATION_PRINTS
   printf( "virtual_unaligned_hot_page::translate virtual_addr, linear_addr -> %p, %p\n", (void*)virtual_addr, (void*)result );
#endif
   return result;
}

new_addr_type virtual_unaligned_hot_page::reverse_translate( new_addr_type unswizzled ) const {
   // Not yet implemented
   assert( 0 );
   return (-1);
}

base_virtual_page* virtual_page_factory::create_new_page( const virtual_page_creation_t& new_page ) {
    base_virtual_page* return_val = NULL;
    switch( new_page.m_type ) {
    case VP_NON_TRANSLATED:
        return_val = new virtual_page( new_page.m_start, new_page.m_end );
        break;
    case VP_TILED:
        return_val = new virtual_tiled_page( new_page.m_start, new_page.m_end, new_page.m_object_size, new_page.m_field_size, new_page.m_tile_width );
        break;
    case VP_HOT_DATA_ALIGNED:
        return_val = new virtual_aligned_hot_page( new_page.m_start, new_page.m_end, new_page.m_object_size, new_page.m_hotbytes, new_page.m_page_size );
        break;
    case VP_HOT_DATA_UNALIGNED:
        return_val = new virtual_unaligned_hot_page( new_page.m_start,
                new_page.m_end,
                new_page.m_object_size,
                new_page.m_hotbytes,
                new_page.m_page_size,
                new_page.m_cache_block_size,
                new_page.m_start_of_first_page );
        break;
    default:
        printf( "virtual_page_factory::create_new_page - Unknown virtual_page_type\n" );
        abort();
    }
    ++m_per_type_pages_created[ new_page.m_type ];
    m_created_pages++;
    return return_val;
}

void warp_inst_t::memory_coalescing_arch_13( bool is_write, mem_access_type access_type )
{
    // see the CUDA manual where it discusses coalescing rules before reading this
    unsigned segment_size = 0;
    unsigned warp_parts = m_config->mem_warp_parts;
    switch( data_size ) {
    case 1: segment_size = 32; break;
    case 2: segment_size = 64; break;
    case 4: case 8: case 16: segment_size = 128; break;
    }
    unsigned subwarp_size = m_config->warp_size / warp_parts;

    for( unsigned subwarp=0; subwarp <  warp_parts; subwarp++ ) {
        std::map<new_addr_type,transaction_info> subwarp_transactions;

        // step 1: find all transactions generated by this subwarp
        for( unsigned thread=subwarp*subwarp_size; thread<subwarp_size*(subwarp+1); thread++ ) {
            if( !active(thread) )
                continue;

            unsigned data_size_coales = data_size;
            unsigned num_accesses = 1;

            if( space.get_type() == local_space || space.get_type() == param_space_local ) {
               // Local memory accesses >4B were split into 4B chunks
               if(data_size >= 4) {
                  data_size_coales = 4;
                  num_accesses = data_size/4;
               }
               // Otherwise keep the same data_size for sub-4B access to local memory
            }


            assert(num_accesses <= MAX_ACCESSES_PER_INSN_PER_THREAD);

            for(unsigned access=0; access<num_accesses; access++) {
                new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[access];
                unsigned block_address = line_size_based_tag_func(addr,segment_size);
                unsigned chunk = (addr&127)/32; // which 32-byte chunk within in a 128-byte chunk does this thread access?
                transaction_info &info = subwarp_transactions[block_address];

                // can only write to one segment
                assert(block_address == line_size_based_tag_func(addr+data_size_coales-1,segment_size));

                info.chunks.set(chunk);
                info.active.set(thread);
                unsigned idx = (addr&127);
                for( unsigned i=0; i < data_size_coales; i++ )
                    info.bytes.set(idx+i);
            }
        }

        // step 2: reduce each transaction size, if possible
        std::map< new_addr_type, transaction_info >::iterator t;
        for( t=subwarp_transactions.begin(); t !=subwarp_transactions.end(); t++ ) {
            new_addr_type addr = t->first;
            const transaction_info &info = t->second;

            memory_coalescing_arch_13_reduce_and_send(is_write, access_type, info, addr, segment_size);

        }
    }
}

void warp_inst_t::memory_coalescing_arch_13_atomic( bool is_write, mem_access_type access_type )
{

   assert(space.get_type() == global_space); // Atomics allowed only for global memory

   // see the CUDA manual where it discusses coalescing rules before reading this
   unsigned segment_size = 0;
   unsigned warp_parts = 2;
   switch( data_size ) {
   case 1: segment_size = 32; break;
   case 2: segment_size = 64; break;
   case 4: case 8: case 16: segment_size = 128; break;
   }
   unsigned subwarp_size = m_config->warp_size / warp_parts;

   for( unsigned subwarp=0; subwarp <  warp_parts; subwarp++ ) {
       std::map<new_addr_type,std::list<transaction_info>> subwarp_transactions; // each block addr maps to a list of transactions

       // step 1: find all transactions generated by this subwarp
       for( unsigned thread=subwarp*subwarp_size; thread<subwarp_size*(subwarp+1); thread++ ) {
           if( !active(thread) )
               continue;

           new_addr_type addr = m_per_scalar_thread[thread].memreqaddr[0];
           unsigned block_address = line_size_based_tag_func(addr,segment_size);
           unsigned chunk = (addr&127)/32; // which 32-byte chunk within in a 128-byte chunk does this thread access?

           // can only write to one segment
           assert(block_address == line_size_based_tag_func(addr+data_size-1,segment_size));

           // Find a transaction that does not conflict with this thread's accesses
           bool new_transaction = true;
           std::list<transaction_info>::iterator it;
           transaction_info* info;
           for(it=subwarp_transactions[block_address].begin(); it!=subwarp_transactions[block_address].end(); it++) {
              unsigned idx = (addr&127);
              if(not it->test_bytes(idx,idx+data_size-1)) {
                 new_transaction = false;
                 info = &(*it);
                 break;
              }
           }
           if(new_transaction) {
              // Need a new transaction
              subwarp_transactions[block_address].push_back(transaction_info());
              info = &subwarp_transactions[block_address].back();
           }
           assert(info);

           info->chunks.set(chunk);
           info->active.set(thread);
           unsigned idx = (addr&127);
           for( unsigned i=0; i < data_size; i++ ) {
               assert(!info->bytes.test(idx+i));
               info->bytes.set(idx+i);
           }
       }

       // step 2: reduce each transaction size, if possible
       std::map< new_addr_type, std::list<transaction_info> >::iterator t_list;
       for( t_list=subwarp_transactions.begin(); t_list !=subwarp_transactions.end(); t_list++ ) {
           // For each block addr
           new_addr_type addr = t_list->first;
           const std::list<transaction_info>& transaction_list = t_list->second;

           std::list<transaction_info>::const_iterator t;
           for(t=transaction_list.begin(); t!=transaction_list.end(); t++) {
               // For each transaction
               const transaction_info &info = *t;
               memory_coalescing_arch_13_reduce_and_send(is_write, access_type, info, addr, segment_size);
           }
       }
   }
}

void warp_inst_t::memory_coalescing_arch_13_reduce_and_send( bool is_write, mem_access_type access_type, const transaction_info &info, new_addr_type addr, unsigned segment_size )
{
   assert( (addr & (segment_size-1)) == 0 );

   const std::bitset<4> &q = info.chunks;
   assert( q.count() >= 1 );
   std::bitset<2> h; // halves (used to check if 64 byte segment can be compressed into a single 32 byte segment)

   unsigned size=segment_size;
   if( segment_size == 128 ) {
       bool lower_half_used = q[0] || q[1];
       bool upper_half_used = q[2] || q[3];
       if( lower_half_used && !upper_half_used ) {
           // only lower 64 bytes used
           size = 64;
           if(q[0]) h.set(0);
           if(q[1]) h.set(1);
       } else if ( (!lower_half_used) && upper_half_used ) {
           // only upper 64 bytes used
           addr = addr+64;
           size = 64;
           if(q[2]) h.set(0);
           if(q[3]) h.set(1);
       } else {
           assert(lower_half_used && upper_half_used);
       }
   } else if( segment_size == 64 ) {
       // need to set halves
       if( (addr % 128) == 0 ) {
           if(q[0]) h.set(0);
           if(q[1]) h.set(1);
       } else {
           assert( (addr % 128) == 64 );
           if(q[2]) h.set(0);
           if(q[3]) h.set(1);
       }
   }
   if( size == 64 ) {
       bool lower_half_used = h[0];
       bool upper_half_used = h[1];
       if( lower_half_used && !upper_half_used ) {
           size = 32;
       } else if ( (!lower_half_used) && upper_half_used ) {
           addr = addr+32;
           size = 32;
       } else {
           assert(lower_half_used && upper_half_used);
       }
   }
   m_accessq.push_back( mem_access_t(access_type,addr,size,is_write,info.active,info.bytes) );
}

void warp_inst_t::completed( unsigned long long cycle ) const 
{
   unsigned long long latency = cycle - issue_cycle; 
   assert(latency <= cycle); // underflow detection 
   ptx_file_line_stats_add_latency(pc, latency * active_count());  
}

base_virtual_page* virtual_page_factory::copy_page( const base_virtual_page* page ) {
    base_virtual_page* return_ptr = NULL;
    if ( page == NULL ) {
        return_ptr = NULL;
    } else if ( dynamic_cast< const virtual_page* >( page ) ) {
        return_ptr = new virtual_page( *dynamic_cast< const virtual_page* >( page ) );
        ++m_per_type_pages_created[ VP_NON_TRANSLATED ];
        m_created_pages++;
    } else if ( dynamic_cast< const virtual_aligned_hot_page* >( page ) ) {
        return_ptr = new virtual_aligned_hot_page( *dynamic_cast< const virtual_aligned_hot_page* >( page ) );
        ++m_per_type_pages_created[ VP_HOT_DATA_ALIGNED ];
        m_created_pages++;
    } else if ( dynamic_cast< const virtual_unaligned_hot_page* >( page ) ) {
        return_ptr = new virtual_unaligned_hot_page( *dynamic_cast< const virtual_unaligned_hot_page* >( page ) );
        ++m_per_type_pages_created[ VP_HOT_DATA_UNALIGNED ];
        m_created_pages++;
    } else if ( dynamic_cast< const virtual_tiled_page* >( page ) ) {
        return_ptr = new virtual_tiled_page( *dynamic_cast< const virtual_tiled_page* >( page ) );
        ++m_per_type_pages_created[ VP_TILED ];
        m_created_pages++;
    } else {
        fprintf( stderr, "Unknown page type\n" );
        abort();
    }
    return return_ptr;
}

vm_page_mapping_payload::vm_page_mapping_payload()
    : m_vm_page_type( VP_NON_TRANSLATED ), m_virtualized_page( NULL ) {}

vm_page_mapping_payload::vm_page_mapping_payload( virtual_page_type config, const virtual_page_creation_t& new_page )
    : m_vm_page_type( config ) {
    m_virtualized_page = virtual_page_factory::get_global_factory()->create_new_page( new_page );
}

vm_page_mapping_payload::vm_page_mapping_payload( virtual_page_type config, const base_virtual_page* page )
    : m_vm_page_type( config ) {
    m_virtualized_page = virtual_page_factory::get_global_factory()->copy_page( page );
}

vm_page_mapping_payload::vm_page_mapping_payload( const vm_page_mapping_payload& to_copy )
    : m_vm_page_type( to_copy.m_vm_page_type ) {
    m_virtualized_page = virtual_page_factory::get_global_factory()->copy_page( to_copy.m_virtualized_page );
}

vm_page_mapping_payload::~vm_page_mapping_payload(){
    virtual_page_factory::get_global_factory()->destroy_page( m_virtualized_page );
}

vm_page_mapping_payload& vm_page_mapping_payload::operator=( const vm_page_mapping_payload& rhs ) {
    m_vm_page_type = rhs.m_vm_page_type;
    virtual_page_factory::get_global_factory()->destroy_page( m_virtualized_page );
    m_virtualized_page = virtual_page_factory::get_global_factory()->copy_page( rhs.m_virtualized_page );
    return *this;
}

virtual_object_divided_page* gpgpu_t::get_virtualized_page( new_addr_type addr ) {
   std::map< new_addr_type, virtual_object_divided_page* >::const_iterator it = m_virtualized_mem_pages.begin();
   for( ; it != m_virtualized_mem_pages.end();++it ) {
      if ( it->second->contains(addr)  ) {
          return it->second;
      }
   }
   return NULL;
}

new_addr_type virtual_tiled_page::translate( const new_addr_type virtual_addr ) const {
    new_addr_type linear_addr = virtual_addr;
    const virtual_access_tile_identifier ident = translate_to_ident( virtual_addr );
    linear_addr = translate( ident );
#if ENABLE_TRANSLATION_PRINTS
    printf( "virtual_tiled_page::translate virtual_addr, linear_addr -> %p, %p\n", (void*)virtual_addr, (void*)linear_addr );
#endif
    return linear_addr;
}

new_addr_type virtual_tiled_page::translate(const virtual_access_tile_identifier &ident) const {
    const int bytes_per_tile = m_tile_width * m_object_size;
    const new_addr_type tile_offset = ident.tile_num * bytes_per_tile;
    const int my_field = ident.offset / m_field_size;
    const new_addr_type field_offset = my_field * m_tile_width * m_field_size;
    const new_addr_type struct_offset = ident.struct_num * m_field_size;
    const new_addr_type physical_addr = (new_addr_type)m_start + tile_offset + field_offset + struct_offset;
    assert( physical_addr < (new_addr_type)m_end );
    return physical_addr;
}

virtual_access_tile_identifier virtual_tiled_page::translate_to_ident( new_addr_type addr ) const {
    virtual_access_tile_identifier result;
    const size_t bytes_from_start = addr - (size_t)m_start;
    const int object_num = bytes_from_start / m_object_size;
    result.tile_num = object_num / m_tile_width;
    result.struct_num = object_num % m_tile_width;
    result.offset = bytes_from_start - object_num * m_object_size;
    return result;
}

new_addr_type virtual_aligned_hot_page::translate( new_addr_type virtual_addr ) const {
    assert( m_page_size % m_object_size == 0 );
    const size_t bytes_from_start = virtual_addr - (size_t)m_start;
    const size_t object_num = bytes_from_start / m_object_size;
    const size_t total_num_object_on_page = m_page_size / m_object_size;

    size_t offset = bytes_from_start - object_num * m_object_size;
    bool am_i_hot = m_hotbytes.size() != 0 ? m_hotbytes.test( offset ) : false;
    // basically we want to know how many hot bytes are you into the hot region.
    unsigned my_hot_offset = 0;
    assert( offset >= 0 );
    assert( m_hotbytes.size() == 0 || (unsigned)offset < m_hotbytes.size() );
    for ( unsigned i = 0; i < m_hotbytes.size(); ++i ) {
       if ( i == (unsigned)offset ) break;
       if ( ( am_i_hot && m_hotbytes.test( i ) )
             || ( !am_i_hot && !m_hotbytes.test( i ) ) ) {
          ++my_hot_offset;
       }
    }
    offset = my_hot_offset;

    assert( ( (size_t)m_end - (size_t)m_start ) % m_object_size == 0 );
    // Have to offset the cold data by all the hot stuff
    const new_addr_type cold_obj_offset = m_hotbytes.count() * total_num_object_on_page;
    const new_addr_type hot_obj_size = m_hotbytes.count();
    const new_addr_type cold_obj_size = m_object_size - hot_obj_size;

    const size_t bytes_per_tile = m_page_size;

    new_addr_type bytes_into_tile;
    if ( am_i_hot ) {
       assert( offset >= 0 && ( unsigned )offset < hot_obj_size );
       bytes_into_tile = object_num * hot_obj_size + offset;
       assert( bytes_into_tile < cold_obj_offset );
    } else {
       bytes_into_tile = object_num * cold_obj_size + offset + cold_obj_offset;
       assert( bytes_into_tile >= cold_obj_offset && bytes_into_tile < bytes_per_tile );
    }

    const new_addr_type linear_addr = (new_addr_type)m_start + bytes_into_tile;
    assert( linear_addr < (new_addr_type)m_end );
#if ENABLE_TRANSLATION_PRINTS
    printf( "virtual_aligned_hot_page::translate virtual_addr, linear_addr -> %p, %p\n", (void*)virtual_addr, (void*)linear_addr );
#endif
    return linear_addr;
}

/*
new_addr_type virtual_aligned_hot_page::translate(const virtual_access_identifier &ident) const {
    assert( ( (size_t)m_end - (size_t)m_start ) % m_object_size == 0 );
    // Have to offset the cold data by all the hot stuff
    const new_addr_type cold_obj_offset = m_hotbytes.count() * m_tile_width;
    const new_addr_type hot_obj_size = m_hotbytes.count();
    const new_addr_type cold_obj_size = m_object_size - hot_obj_size;

    const size_t bytes_per_tile = m_tile_width * m_object_size;
    const new_addr_type tile_offset = ident.tile_num * bytes_per_tile;

    new_addr_type bytes_into_tile;
    if ( ident.am_i_hot ) {
       assert( ident.offset >= 0 && ( unsigned )ident.offset < hot_obj_size );
       bytes_into_tile = ident.struct_num * hot_obj_size + ident.offset;
       assert( bytes_into_tile < cold_obj_offset );
    } else {
       assert( (unsigned)ident.offset <= cold_obj_offset );
       bytes_into_tile = ident.struct_num * cold_obj_size + ident.offset + cold_obj_offset;
       assert( bytes_into_tile >= cold_obj_offset && bytes_into_tile < bytes_per_tile );
    }

    const new_addr_type physical_addr = (new_addr_type)m_start + tile_offset + bytes_into_tile;
    //printf( "virtualized_tiled_region::translate\nident.am_i_hot = %d\nm_start = %p\ntile_offset=%llu\nbytes_into_tile=%llu\nhot_obj_size=%llu\nident.struct_num=%d\nident.offset=%d\n", ident.am_i_hot, m_start, tile_offset, bytes_into_tile, hot_obj_size, ident.struct_num, ident.offset );
    assert( physical_addr < (new_addr_type)m_end );

    return physical_addr;
}

// Take in an identifier and translates it back into unmapped space
new_addr_type virtual_aligned_hot_page::reverse_translate(const virtual_access_identifier &ident) const {
   //assert( ( (size_t)m_end - (size_t)m_start ) % m_object_size == 0 );

   const size_t bytes_per_tile = m_tile_width * m_object_size;
   const new_addr_type tile_offset = ident.tile_num * bytes_per_tile;
   const new_addr_type struct_offset = m_object_size * ident.struct_num;

   new_addr_type in_obj_offset = 0;
   unsigned hot_bytes_in = 0;
   for ( unsigned i = 0; i < m_hotbytes.size(); ++i ) {
      if ( ( ident.am_i_hot && m_hotbytes.test( i ) ) || ( !ident.am_i_hot && !m_hotbytes.test( i ) ) ) {
         if ( hot_bytes_in == (unsigned)ident.offset ) break;
         ++hot_bytes_in;
      }
      ++in_obj_offset;
   }

   const new_addr_type unmapped_addr = (new_addr_type)m_start + tile_offset + struct_offset + in_obj_offset;
   assert( unmapped_addr < (new_addr_type)m_end );

   //printf( "Reverse Translation of ident.am_i_hot=%d\nident.offset=%d\nident.struct_num=%d\nident.tile_num=%d\nLeads to %p\n",
     //    ident.am_i_hot, ident.offset, ident.struct_num, ident.tile_num, (void*)unmapped_addr );
   //return unmapped_addr;
}

virtual_access_identifier virtual_aligned_hot_page::reverse_translate( new_addr_type addr ) const {
   virtual_access_identifier result;
   const new_addr_type cold_obj_offset = get_hot_data_size() * get_tile_width();
   const new_addr_type hot_obj_size = get_hot_data_size();
   const new_addr_type cold_obj_size = get_object_size() - hot_obj_size;
   const size_t bytes_per_tile = get_tile_width() * get_object_size();
   result.tile_num = ( addr - (new_addr_type)get_start_addr() ) / bytes_per_tile;
   const new_addr_type my_tile_start = (new_addr_type)get_start_addr() + result.tile_num * bytes_per_tile;
   const new_addr_type my_tile_offset = addr - my_tile_start;
   result.am_i_hot = ( addr - my_tile_start ) < ( hot_obj_size * get_tile_width() );
   result.struct_num = result.am_i_hot ?
            ( my_tile_offset ) / hot_obj_size :
            ( my_tile_offset - cold_obj_offset ) / cold_obj_size;
   result.offset = result.am_i_hot ?
         my_tile_offset - result.struct_num * hot_obj_size :
         my_tile_offset - cold_obj_offset - result.struct_num * cold_obj_size;
   //printf( "Deconstruction of linear address %p\nLeads to tile=%d\nobject=%d\noffset=%d\nresult.am_i_hot=%d\n", (void*)addr, result.tile_num, result.struct_num, result.offset, result.am_i_hot );
   return result;
}

// TODO - tgrogers Tiling is no longer working with this hot data stuff.  Need an options to enable either tiling or hot data remapping.
//          Come back and fix this
virtual_access_identifier virtual_aligned_hot_page::translate( new_addr_type addr ) const {
   virtual_access_identifier result;
   const size_t bytes_from_start = addr - (size_t)m_start;
   const int object_num = bytes_from_start / m_object_size;

   result.tile_num = object_num / m_tile_width;
   result.struct_num = object_num % m_tile_width;
   result.offset = bytes_from_start - object_num * m_object_size;
   result.am_i_hot = m_hotbytes.size() != 0 ? m_hotbytes.test( result.offset ) : false;
   result.orig_addr=addr;
   // basically we want to know how many hot bytes are you into the hot region.
   unsigned my_hot_offset = 0;
   assert( result.offset >= 0 );
   assert( m_hotbytes.size() == 0 || (unsigned)result.offset < m_hotbytes.size() );
   for ( unsigned i = 0; i < m_hotbytes.size(); ++i ) {
      if ( i == (unsigned)result.offset ) break;
      if ( ( result.am_i_hot && m_hotbytes.test( i ) )
            || ( !result.am_i_hot && !m_hotbytes.test( i ) ) ) {
         ++my_hot_offset;
      }
   }
   result.offset = my_hot_offset;
   //printf("VIRTUALIZED addr=0x%llx it->object_size=%d object_num=%d vm_ident.tile_num=%d vm_ident.struct_num=%d vm_ident.offset=%d\n",
    //   addr, it->object_size, object_num, vm_ident.tile_num, vm_ident.struct_num, vm_ident.offset);
   return result;
}
*/
extern size_t g_hack_start_of_static_space;
extern size_t g_hack_end_of_static_space;

void gpgpu_t::print_virtualized_pages( FILE* fp, bool print_all_pages ) {
   std::map< new_addr_type, virtual_object_divided_page* >::const_iterator it = m_virtualized_mem_pages.begin();
   size_t total_mapped_size = 0;
   size_t total_should_be_static_mapped_size = 0;
   fprintf( fp, "You have mapped %zu regions.\n", m_virtualized_mem_pages.size() );
   for( ; it != m_virtualized_mem_pages.end();++it ) {
      assert( it->second->get_end_addr() > it->second->get_start_addr() );
      if ( (size_t)it->second->get_start_addr() >= g_hack_start_of_static_space && (size_t)it->second->get_end_addr() <= g_hack_end_of_static_space ) {
         total_should_be_static_mapped_size += ( (size_t)it->second->get_end_addr() - (size_t)it->second->get_start_addr() );
      }
      if ( print_all_pages ) {
          it->second->print( stdout );
      }
      total_mapped_size += ( (size_t)it->second->get_end_addr() - (size_t)it->second->get_start_addr() );
   }
   fprintf( fp, "You have mapped %zu bytes of memory - %zu bytes from the static region. Percent of static_region=%f\n"
         , total_mapped_size, total_should_be_static_mapped_size, (float)total_should_be_static_mapped_size / (float)(g_hack_end_of_static_space - g_hack_start_of_static_space) );
}

size_t g_hack_start_of_static_space;
size_t g_hack_end_of_static_space;
void gpgpu_t::handle_static_virtual_page( void* start_of_vm,
            size_t total_size,
            size_t field_size,
            size_t tile_width,
            size_t object_size,
            size_t page_size,
            boost::dynamic_bitset<> hotbytes ) {
    g_hack_start_of_static_space = (size_t)start_of_vm;
    g_hack_end_of_static_space = (size_t)start_of_vm + total_size;

    const size_t tile_size = object_size * tile_width;
    const size_t number_of_tiles = total_size / tile_size;

    virtual_page_creation_t new_page;
    new_page.m_type = g_translated_page_type;
    new_page.m_object_size = object_size;
    switch ( g_translated_page_type ) {
    case VP_NON_TRANSLATED:
        // Intentional Fall Through
    case VP_HOT_DATA_UNALIGNED:
        printf( "Virtual Memory space not declared, only tiled and hot aligned hot data can be statically defined\n" );
        return;
    case VP_TILED: {
        new_page.m_start = start_of_vm;
        new_page.m_end = (char*)start_of_vm + total_size;
        new_page.m_field_size = field_size;
        new_page.m_tile_width = tile_width;
        base_virtual_page* page = get_page_factory().create_new_page( new_page );
        if ( total_size != number_of_tiles * tile_size ) {
            fprintf( stderr, "ERROR gpgpu_t::handle_static_virtual_page: Virtual memory space requires a tile alignmed memory space, the total_size=%zu was passed in.  Please ensure a multiple of %zu\n"
                    , total_size, tile_size );
            abort();
        }
        page->print(stdout);
        set_static_virtualized_page( dynamic_cast< virtual_object_divided_page* >( page ) );
    } break;
    case VP_HOT_DATA_ALIGNED: {
        assert( total_size % object_size == 0 );
        assert( page_size % object_size == 0 && page_size > object_size );
        new_page.m_hotbytes = hotbytes;
        new_page.m_page_size = page_size;
        for ( new_addr_type current_page = ( new_addr_type )start_of_vm; current_page < ( new_addr_type )start_of_vm + total_size; current_page += page_size ) {
            new_page.m_start = (void*)current_page;
            new_page.m_end = (void*)( current_page + page_size );
            base_virtual_page* page = get_page_factory().create_new_page( new_page );
            set_static_virtualized_page( dynamic_cast< virtual_object_divided_page* >( page ) );
        }
    } break;
    default:
        printf( "Unkown g_translated_page_type\n" );
        abort();
    }
}

void gpgpu_t::set_static_virtualized_page( virtual_object_divided_page* page ) {
    if ( STATIC_TRANSLATION == g_translation_config ) {
        set_page( page );
    } else {
        m_page_factory.destroy_page( page );
    }
}

void gpgpu_t::set_dynamic_virtualized_page( virtual_object_divided_page* page ) {
    if ( DYNAMIC_TRANSLATION == g_translation_config ) {
        set_page( page );
    } else {
        m_page_factory.destroy_page( page );
    }
}

void gpgpu_t::set_page( virtual_object_divided_page* page ) {
    assert( page );
    if ( m_virtualized_mem_pages.find( ( new_addr_type )page->get_start_addr() ) != m_virtualized_mem_pages.end()
            && *page == *m_virtualized_mem_pages.find( ( new_addr_type )page->get_start_addr() )->second ) {
        m_page_factory.destroy_page( page );
    } else {
        m_virtualized_mem_pages[ ( new_addr_type )page->get_start_addr() ] = page;
    }
}

void gpgpu_t::remove_virtualized_page( new_addr_type addr ) {
    std::map< new_addr_type, virtual_object_divided_page* >::iterator page = m_virtualized_mem_pages.find( addr );
    if ( page != m_virtualized_mem_pages.end() ) {
        m_page_factory.destroy_page( page->second );
        m_virtualized_mem_pages.erase( addr );
    }
}

unsigned kernel_info_t::m_next_uid = 1;

kernel_info_t::kernel_info_t( dim3 gridDim, dim3 blockDim, class function_info *entry )
{
    m_kernel_entry=entry;
    m_grid_dim=gridDim;
    m_block_dim=blockDim;
    m_next_cta.x=0;
    m_next_cta.y=0;
    m_next_cta.z=0;
    m_next_tid=m_next_cta;
    m_num_cores_running=0;
    m_uid = m_next_uid++;
    m_param_mem = new memory_space_impl<8192>("param",64*1024);
}

kernel_info_t::~kernel_info_t()
{
    assert( m_active_threads.empty() );
    delete m_param_mem;
}

std::string kernel_info_t::name() const
{
    return m_kernel_entry->get_name();
}

simt_stack::simt_stack( unsigned wid, unsigned warpSize)
{
    m_warp_id=wid;
    m_warp_size = warpSize;
    m_stack_top = 0;
    m_pc = (address_type*)calloc(m_warp_size * 2, sizeof(address_type));
    m_calldepth = (unsigned int*)calloc(m_warp_size * 2, sizeof(unsigned int));
    m_active_mask = new simt_mask_t[m_warp_size * 2];
    m_recvg_pc = (address_type*)calloc(m_warp_size * 2, sizeof(address_type));
    m_branch_div_cycle = (unsigned long long *)calloc(m_warp_size * 2, sizeof(unsigned long long ));
    reset();
}

void simt_stack::reset()
{
    m_stack_top = 0;
    memset(m_pc, -1, m_warp_size * 2 * sizeof(address_type));
    memset(m_calldepth, 0, m_warp_size * 2 * sizeof(unsigned int));
    memset(m_recvg_pc, -1, m_warp_size * 2 * sizeof(address_type));
    memset(m_branch_div_cycle, 0, m_warp_size * 2 * sizeof(unsigned long long ));
    for( unsigned i=0; i < 2*m_warp_size; i++ ) 
        m_active_mask[i].reset();
}

void simt_stack::launch( address_type start_pc, const simt_mask_t &active_mask )
{
    reset();
    m_pc[0] = start_pc;
    m_calldepth[0] = 1;
    m_active_mask[0] = active_mask;
}

const simt_mask_t &simt_stack::get_active_mask() const
{
    return m_active_mask[m_stack_top];
}

void simt_stack::get_pdom_stack_top_info( unsigned *pc, unsigned *rpc ) const
{
   *pc = m_pc[m_stack_top];
   *rpc = m_recvg_pc[m_stack_top];
}

unsigned simt_stack::get_rp() const 
{ 
    return m_recvg_pc[m_stack_top]; 
}

void simt_stack::print (FILE *fout) const
{
    const simt_stack *warp=this;
    for ( unsigned k=0; k <= warp->m_stack_top; k++ ) {
        if ( k==0 ) {
            fprintf(fout, "w%02d %1u ", m_warp_id, k );
        } else {
            fprintf(fout, "    %1u ", k );
        }
        for (unsigned j=0; j<m_warp_size; j++)
            fprintf(fout, "%c", (warp->m_active_mask[k].test(j)?'1':'0') );
        fprintf(fout, " pc: 0x%03x", warp->m_pc[k] );
        if ( warp->m_recvg_pc[k] == (unsigned)-1 ) {
            fprintf(fout," rp: ---- cd: %2u ", warp->m_calldepth[k] );
        } else {
            fprintf(fout," rp: %4u cd: %2u ", warp->m_recvg_pc[k], warp->m_calldepth[k] );
        }
        if ( warp->m_branch_div_cycle[k] != 0 ) {
            fprintf(fout," bd@%6u ", (unsigned) warp->m_branch_div_cycle[k] );
        } else {
            fprintf(fout," " );
        }
        ptx_print_insn( warp->m_pc[k], fout );
        fprintf(fout,"\n");
    }
}

void simt_stack::update( simt_mask_t &thread_done, addr_vector_t &next_pc, address_type recvg_pc ) 
{
    int stack_top = m_stack_top;

    assert( next_pc.size() == m_warp_size );

    simt_mask_t  top_active_mask = m_active_mask[stack_top];
    address_type top_recvg_pc = m_recvg_pc[stack_top];
    address_type top_pc = m_pc[stack_top]; // the pc of the instruction just executed 

    assert(top_active_mask.any());

    const address_type null_pc = 0;
    bool warp_diverged = false;
    address_type new_recvg_pc = null_pc;
    while (top_active_mask.any()) {

        // extract a group of threads with the same next PC among the active threads in the warp
        address_type tmp_next_pc = null_pc;
        simt_mask_t tmp_active_mask;
        for (int i = m_warp_size - 1; i >= 0; i--) {
            if ( top_active_mask.test(i) ) { // is this thread active?
                if (thread_done.test(i)) {
                    top_active_mask.reset(i); // remove completed thread from active mask
                } else if (tmp_next_pc == null_pc) {
                    tmp_next_pc = next_pc[i];
                    tmp_active_mask.set(i);
                    top_active_mask.reset(i);
                } else if (tmp_next_pc == next_pc[i]) {
                    tmp_active_mask.set(i);
                    top_active_mask.reset(i);
                }
            }
        }

        // discard the new entry if its PC matches with reconvergence PC
        // that automatically reconverges the entry
        if (tmp_next_pc == top_recvg_pc) continue;

        // this new entry is not converging
        // if this entry does not include thread from the warp, divergence occurs
        if (top_active_mask.any() && !warp_diverged ) {
            warp_diverged = true;
            // modify the existing top entry into a reconvergence entry in the pdom stack
            new_recvg_pc = recvg_pc;
            if (new_recvg_pc != top_recvg_pc) {
                m_pc[stack_top] = new_recvg_pc;
                m_branch_div_cycle[stack_top] = gpu_sim_cycle;
                stack_top += 1;
                m_branch_div_cycle[stack_top] = 0;
            }
        }

        // discard the new entry if its PC matches with reconvergence PC
        if (warp_diverged && tmp_next_pc == new_recvg_pc) continue;

        // update the current top of pdom stack
        m_pc[stack_top] = tmp_next_pc;
        m_active_mask[stack_top] = tmp_active_mask;
        if (warp_diverged) {
            m_calldepth[stack_top] = 0;
            m_recvg_pc[stack_top] = new_recvg_pc;
        } else {
            m_recvg_pc[stack_top] = top_recvg_pc;
        }
        stack_top += 1; // set top to next entry in the pdom stack
    }
    m_stack_top = stack_top - 1;

    assert(m_stack_top >= 0);
    assert(m_stack_top < m_warp_size * 2);

    if (warp_diverged) {
        ptx_file_line_stats_add_warp_divergence(top_pc, 1); 
    }
}

void core_t::execute_warp_inst_t(warp_inst_t &inst, unsigned warpSize, unsigned warpId){
    for ( unsigned t=0; t < warpSize; t++ ) {
        if( inst.active(t) ) {
        if(warpId==(unsigned (-1)))
            warpId = inst.warp_id();
        unsigned tid=warpSize*warpId+t;
        m_thread[tid]->ptx_exec_inst(inst,t);
        
        //virtual function
        checkExecutionStatusAndUpdate(inst,t,tid);
        }
    } 
}
  
bool  core_t::ptx_thread_done( unsigned hw_thread_id ) const  
{
    return ((m_thread[ hw_thread_id ]==NULL) || m_thread[ hw_thread_id ]->is_done());
}
  
void core_t::updateSIMTStack(unsigned warpId, unsigned warpSize, warp_inst_t * inst){
simt_mask_t thread_done;
addr_vector_t next_pc;
unsigned wtid = warpId * warpSize;
for (unsigned i = 0; i < warpSize; i++) {
    if( ptx_thread_done(wtid+i) ) {
         thread_done.set(i);
         next_pc.push_back( (address_type)-1 );
     } else {
         if( inst->reconvergence_pc == RECONVERGE_RETURN_PC ) 
             inst->reconvergence_pc = get_return_pc(m_thread[wtid+i]);
         next_pc.push_back( m_thread[wtid+i]->get_pc() );
     }
}
m_simt_stack[warpId]->update(thread_done,next_pc,inst->reconvergence_pc);
}

//! Get the warp to be executed using the data taken form the SIMT stack
warp_inst_t core_t::getExecuteWarp(unsigned warpId){
    unsigned pc,rpc;
    m_simt_stack[warpId]->get_pdom_stack_top_info(&pc,&rpc);
    warp_inst_t wi= *ptx_fetch_inst(pc);
    wi.set_active(m_simt_stack[warpId]->get_active_mask());
    return wi;
}

void core_t::initilizeSIMTStack(unsigned warps, unsigned warpsSize)
{ 
m_simt_stack = new simt_stack*[warps];
    for (unsigned i = 0; i < warps; ++i) 
        m_simt_stack[i] = new simt_stack(i,warpsSize);
}
bool hack_use_warp_prot_list = false;
