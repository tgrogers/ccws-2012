// Copyright (c) 2009-2011, Tor M. Aamodt
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <list>
#include <set>

#include "../option_parser.h"
#include "mem_fetch.h"
#include "dram.h"
#include "gpu-cache.h"
#include "histogram.h"
#include "l2cache.h"
#include "../intersim/statwraper.h"
#include "../abstract_hardware_model.h"
#include "gpu-sim.h"
#include "shader.h"
#include "mem_latency_stat.h"


mem_fetch * partition_mf_allocator::alloc(new_addr_type addr, mem_access_type type, unsigned size, bool wr, mem_fetch_payload *payload ) const 
{
    assert( wr );
    mem_access_t access( type, addr, size, wr );
    mem_fetch *mf = new mem_fetch( access, 
                                   NULL,
                                   WRITE_PACKET_SIZE, 
                                   -1, 
                                   -1, 
                                   -1,
                                   m_memory_config,
                                   payload );
    return mf;
}

mem_fetch * partition_mf_allocator::alloc(unsigned sid, unsigned cluster_id, new_addr_type addr, mem_access_type type, unsigned size, bool wr, mem_fetch_payload *payload ) const
{
    assert( dynamic_cast< vm_exploded_L2_payload* >( payload ) );
    //assert( !wr );
    mem_access_t access( type, addr, size, wr );
    mem_fetch *mf = new mem_fetch( access,
                                   NULL,
                                   READ_PACKET_SIZE,
                                   -1,
                                   sid,
                                   cluster_id,
                                   m_memory_config,
                                   payload );
    return mf;
}

mem_fetch *partition_mf_allocator::alloc( unsigned sid, unsigned cluster_id, new_addr_type addr, mem_access_type type, mem_fetch_payload *payload ) const
{
    mem_access_t access( type, addr, payload->size(), false );
    mem_fetch *mf = new mem_fetch( access, 
                                   NULL,
                                   READ_PACKET_SIZE, 
                                   -1, 
                                   sid, 
                                   cluster_id,
                                   m_memory_config,
                                   payload );
    return mf;
}

memory_partition_unit::memory_partition_unit( unsigned partition_id, 
                                              const struct memory_config *config,
                                              class memory_stats_t *stats )
    : m_vm_region_cache( NULL )
{
    m_id = partition_id;
    m_config=config;
    m_stats=stats;
    m_dram = new dram_t(m_id,m_config,m_stats,this);

    char L2c_name[32];
    snprintf(L2c_name, 32, "L2_bank_%03d", m_id);
    m_L2interface = new L2interface(this);
    m_mf_allocator = new partition_mf_allocator(config);

    if(!m_config->m_L2_config.disabled()) {
        switch ( m_config->m_L2_config.get_cache_type() ) {
        case CACHE_NORMAL:
            m_L2cache = new data_cache(L2c_name,m_config->m_L2_config,-1,-1,m_L2interface,m_mf_allocator,IN_PARTITION_L2_MISS_QUEUE, false);
            break;
        case CACHE_SECTORED:
            m_L2cache = new distill_cache(L2c_name,m_config->m_L2_config,-1,-1,m_L2interface,m_mf_allocator,IN_PARTITION_L2_MISS_QUEUE, false);
            break;
        case CACHE_VIRTUAL_POLICY_MANAGED:
            m_L2cache = new virtual_policy_cache(L2c_name,m_config->m_L2_config,-1,-1,m_L2interface,m_mf_allocator,IN_PARTITION_L2_MISS_QUEUE, false);
            m_vm_region_cache = new region_cache(m_config->m_vm_region_config);
            break;
        default:
            printf("Unknown L2 cache type\n");
            abort();
        }
    }

    unsigned int icnt_L2;
    unsigned int L2_dram;
    unsigned int dram_L2;
    unsigned int L2_icnt;

    sscanf(m_config->gpgpu_L2_queue_config,"%u:%u:%u:%u", &icnt_L2,&L2_dram,&dram_L2,&L2_icnt );
    m_icnt_L2_queue = new fifo_pipeline<mem_fetch>("icnt-to-L2",0,icnt_L2); 
    m_L2_dram_queue = new fifo_pipeline<mem_fetch>("L2-to-dram",0,L2_dram);
    m_dram_L2_queue = new fifo_pipeline<mem_fetch>("dram-to-L2",0,dram_L2);
    m_L2_icnt_queue = new fifo_pipeline<mem_fetch>("L2-to-icnt",0,L2_icnt);
    wb_addr=-1;

	// Currently static region size
    m_region_metric = new region_metric(config->m_L2_config.get_line_sz(), 4096);

}

memory_partition_unit::~memory_partition_unit()
{
    delete m_icnt_L2_queue;
    delete m_L2_dram_queue;
    delete m_dram_L2_queue;
    delete m_L2_icnt_queue;
    delete m_L2cache;
    delete m_L2interface;
    delete m_vm_region_cache;
}


void memory_partition_unit::cache_cycle( unsigned cycle )
{
    // L2 fill responses 
    if( !m_config->m_L2_config.disabled()) {
        if ( m_L2cache->access_ready() && !m_L2_icnt_queue->full() ) {
            mem_fetch *mf = m_L2cache->next_access();
            mf->set_all_valid_words();
            mf->set_reply();
            mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
            m_L2_icnt_queue->push(mf);
        }
    }

    // DRAM to L2 (texture) and icnt (not texture)
    if ( !m_dram_L2_queue->empty() ) {
        mem_fetch *mf = m_dram_L2_queue->top();

        // When we are receiving an exploded response, we need to wait for all the pieces, then send back the original memory request
        vm_exploded_L2_payload* exploded_pay = dynamic_cast< vm_exploded_L2_payload* >( mf->get_payload() );
        if ( exploded_pay ) {
           if ( exploded_pay->get_exploded_in_flight() > 1 ) {
              exploded_pay->decrement_exploded_in_flight();
              m_dram_L2_queue->pop();
              delete mf;
              return;
           } else {
              mem_fetch* original = exploded_pay->get_original_fetch();
              assert( original );
              mf = original;
           }
        }

        if ( !m_config->m_L2_config.disabled() && m_L2cache->waiting_for_fill(mf) ) {
            mf->set_all_valid_words();
            mf->set_status(IN_PARTITION_L2_FILL_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
            m_L2cache->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle);

            m_dram_L2_queue->pop();
        } else if ( !m_L2_icnt_queue->full() ) {
            mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
            mf->set_all_valid_words();
            m_L2_icnt_queue->push(mf);
            m_dram_L2_queue->pop();
        }
    }

    // prior L2 misses inserted into m_L2_dram_queue here
    if( !m_config->m_L2_config.disabled() )
       m_L2cache->cycle();

    // new L2 texture accesses and/or non-texture accesses
    if ( !m_L2_dram_queue->full() && !m_icnt_L2_queue->empty() ) {
        mem_fetch *mf = m_icnt_L2_queue->top();

        if( mf->get_access_type() == VM_POLICY_CHANGE_REQ ) {
            region_cache_policy_update(mf);
        } else if ( !m_config->m_L2_config.disabled() &&
              ( (m_config->m_L2_texure_only && mf->istexture()) || (!m_config->m_L2_texure_only) )
           ) {
            bool do_access = (m_config->m_vm_region_config.enabled() && ( m_L2_icnt_queue->get_length()+2 < m_L2_icnt_queue->get_max_len() )) ||
                  ( !m_config->m_vm_region_config.enabled() && !m_L2_icnt_queue->full() );
            if( do_access ) {
                region_cache_policy_check(mf);

                std::list<cache_event> events;
                mf->set_L2_access(true);	// Tells the data cache access that this request is from the L2 cache -> Mem_divergence_metrics
                enum cache_request_status status = m_L2cache->access(/*tgrogers - temp*/mf->get_addr(),mf,gpu_sim_cycle+gpu_tot_sim_cycle,events);
                mf->set_L2_access(false);

                bool write_sent = was_write_sent(events);
                bool read_sent = was_read_sent(events);

                if ( status == HIT ) {
                    if( !write_sent ) {
                        // L2 cache replies
                        assert(!read_sent);
                        if( mf->get_access_type() == L1_WRBK_ACC ) {
                            m_request_tracker.erase(mf);
                            delete mf;
                        } else {
                            mf->set_reply();
                            mf->set_status(IN_PARTITION_L2_TO_ICNT_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
                            m_L2_icnt_queue->push(mf);
                        }
                        m_icnt_L2_queue->pop();
                    } else {
                        assert(write_sent);
                        m_icnt_L2_queue->pop();
                    }
                } else if ( status != RESERVATION_FAIL ) {
                    // L2 cache accepted request
                    m_icnt_L2_queue->pop();
                } else {
                    assert(!write_sent);
                    assert(!read_sent);
                    // L2 cache lock-up: will try again next cycle
                }
            }
        } else {
            // L2 is disabled or non-texture access to texture-only L2
            mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
            m_L2_dram_queue->push(mf);
            m_icnt_L2_queue->pop();
        }
    }

    // ROP delay queue
    if( !m_rop.empty() && (cycle >= m_rop.front().ready_cycle) && !m_icnt_L2_queue->full() ) {
        mem_fetch* mf = m_rop.front().req;
        m_rop.pop();
        m_icnt_L2_queue->push(mf);
        mf->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
    }
}

bool memory_partition_unit::full() const
{
    return m_icnt_L2_queue->full();
}

void memory_partition_unit::print_cache_stat(unsigned &accesses, unsigned &misses) const
{
    FILE *fp = stdout;
    if( !m_config->m_L2_config.disabled() )
       m_L2cache->print(fp,accesses,misses);
}

void memory_partition_unit::print( FILE *fp ) const
{
    if ( !m_request_tracker.empty() ) {
        fprintf(fp,"Memory Parition %u: pending memory requests:\n", m_id);
        for ( std::set<mem_fetch*>::const_iterator r=m_request_tracker.begin(); r != m_request_tracker.end(); ++r ) {
            mem_fetch *mf = *r;
            if ( mf )
                mf->print(fp);
            else
                fprintf(fp," <NULL mem_fetch?>\n");
        }
    }
    if( !m_config->m_L2_config.disabled() )
       m_L2cache->display_state(fp);
    m_dram->print(fp); 
}

void memory_stats_t::print( FILE *fp )
{
    fprintf(fp,"L2_write_miss = %d\n", L2_write_miss);
    fprintf(fp,"L2_write_hit = %d\n", L2_write_hit);
    fprintf(fp,"L2_read_miss = %d\n", L2_read_miss);
    fprintf(fp,"L2_read_hit = %d\n", L2_read_hit);

    fprintf(fp,"VM region swizzled, request default = %u\n", vm_region_swizzled_request_default );
    fprintf(fp,"VM region default,  request swizzled = %u\n", vm_region_default_request_swizzled );
    fprintf(fp,"VM region swizzled, request different swizzled = %u\n", vm_region_swizzled_request_swizzled );
    fprintf(fp,"VM region eviction other = %u\n", vm_region_evict_other );
    fprintf(fp,"VM region eviction (change swizzle) = %u\n", vm_region_evict_same_change_swizzle );
    fprintf(fp,"VM region eviction (change to default) = %u\n", vm_region_evict_same_change_to_default );
}

void memory_stats_t::visualizer_print( gzFile visualizer_file )
{
   gzprintf(visualizer_file, "Ltwowritemiss: %d\n", L2_write_miss);
   gzprintf(visualizer_file, "Ltwowritehit: %d\n",  L2_write_hit);
   gzprintf(visualizer_file, "Ltworeadmiss: %d\n", L2_read_miss);
   gzprintf(visualizer_file, "Ltworeadhit: %d\n", L2_read_hit);
   if (num_mfs) 
      gzprintf(visualizer_file, "averagemflatency: %lld\n", mf_total_lat/num_mfs);
}

void gpgpu_sim::L2c_print_cache_stat() const
{
    unsigned i, j, k;
    for (i=0,j=0,k=0;i<m_memory_config->m_n_mem;i++)
        m_memory_partition_unit[i]->print_cache_stat(k,j);
    printf("L2_Cache_Total_Miss_Rate = %0.3f\n", (float)j/k);
}

unsigned memory_partition_unit::flushL2() 
{ 
    m_L2cache->flush(); 
    return 0; // L2 is read only in this version
}

bool memory_partition_unit::busy() const 
{
    return !m_request_tracker.empty();
}

void memory_partition_unit::push( mem_fetch* req, unsigned long long cycle ) 
{
    if (req) {
        m_request_tracker.insert(req);
        m_stats->memlatstat_icnt2mem_pop(req);
        if( req->istexture() ) {
            m_icnt_L2_queue->push(req);
            req->set_status(IN_PARTITION_ICNT_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
        } else {
            rop_delay_t r;
            r.req = req;
            r.ready_cycle = cycle + m_config->rop_latency;
            m_rop.push(r);
            req->set_status(IN_PARTITION_ROP_DELAY,gpu_sim_cycle+gpu_tot_sim_cycle);
        }
    }
}

mem_fetch* memory_partition_unit::pop() 
{
    mem_fetch* mf = m_L2_icnt_queue->pop();
    m_request_tracker.erase(mf);
    if ( mf && mf->isatomic() )
        mf->do_atomic();
    if( mf && (mf->get_access_type() == L2_WRBK_ACC || mf->get_access_type() == L1_WRBK_ACC) ) {
        delete mf;
        mf = NULL;
    } 
    return mf;
}

mem_fetch* memory_partition_unit::top() 
{
    mem_fetch *mf = m_L2_icnt_queue->top();
    if( mf && (mf->get_access_type() == L2_WRBK_ACC || mf->get_access_type() == L1_WRBK_ACC) ) {
        m_L2_icnt_queue->pop();
        m_request_tracker.erase(mf);
        delete mf;
        mf = NULL;
    } 
    return mf;
}

void memory_partition_unit::set_done( mem_fetch *mf )
{
    m_request_tracker.erase(mf);
}

void memory_partition_unit::dram_cycle() 
{ 
    // pop completed memory request from dram and push it to dram-to-L2 queue 
    if ( !m_dram_L2_queue->full() ) {
        mem_fetch* mf = m_dram->pop();
        if (mf) {
            if( mf->get_access_type() == L1_WRBK_ACC ) {
                m_request_tracker.erase(mf);
                delete mf;
            } else {
                m_dram_L2_queue->push(mf);
                mf->set_status(IN_PARTITION_DRAM_TO_L2_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
            }
        }
    }
    m_dram->cycle(); 
    m_dram->dram_log(SAMPLELOG);   

    if( !m_dram->full() && !m_L2_dram_queue->empty() ) {
        // L2->DRAM queue to DRAM latency queue
        mem_fetch *mf = m_L2_dram_queue->pop();
        dram_delay_t d;
        d.req = mf;
        d.ready_cycle = gpu_sim_cycle+gpu_tot_sim_cycle + m_config->dram_latency;
        m_dram_latency_queue.push(d);
        mf->set_status(IN_PARTITION_DRAM_LATENCY_QUEUE,gpu_sim_cycle+gpu_tot_sim_cycle);
    }

    // DRAM latency queue
    if( !m_dram_latency_queue.empty() && ( (gpu_sim_cycle+gpu_tot_sim_cycle) >= m_dram_latency_queue.front().ready_cycle ) && !m_dram->full() ) {
        mem_fetch* mf = m_dram_latency_queue.front().req;
        m_dram_latency_queue.pop();
        m_dram->push(mf);
    }
}


void memory_partition_unit::region_cache_policy_update( mem_fetch *mf )
{
    if( m_L2_icnt_queue->full() ) 
        return; // can't send ack
    
    // cases:
    // 1. region cache is default (miss) and request is default => nothing to do, ack
    // 2. region cache is default (miss) and request is swizzle
    //    (a) adding region causes eviction of another region => evict *other* region from L2, update region cache, ack
    //    (b) adding region does not evict another region => update region cache, ack
    // 3. region cache is swizzled (hit) and request is default => evict region from L2, evict mapping from region cache, ack
    // 4. region cache is swizzled (hit) and request is swizzled 
    //    (a) swizzle parameters match => nothing to do, ack
    //    (b) swizzle parameters differ => evict from L2, update region cache, ack

    vm_page_mapping_payload *payload = dynamic_cast<vm_page_mapping_payload*>(mf->get_payload());
    virtual_page_type request_format = payload->get_page_type();
    vm_page_mapping_payload region_policy;
    enum cache_request_status rc_status = m_vm_region_cache->access(mf,gpu_sim_cycle+gpu_tot_sim_cycle,region_policy);

    if ( rc_status == MISS ) {
        // region is mapped as default (no translation) in cache 
        switch ( request_format ) {
        case NO_TRANSLATION: 
            // 1. region cache is default (miss) and request is default => nothing to do, ack
            break;
        case DYNAMIC_TRANSLATION: {
                // 2. region cache is default (miss) and request is swizzle
                bool evicted=false;
                vm_page_mapping_payload evicted_policy;
                m_vm_region_cache->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle,*payload,evicted,evicted_policy);
                if ( evicted ) {
                    //    (a) adding region causes eviction of another region => evict *other* region from L2, update region cache, ack
                    m_stats->vm_region_evict_other++;
                    // TODO: model evicting of other region from L2
                } else {
                    //    (b) adding region does not evict another region => update region cache, ack
                }
                break;}
        default: abort();
        }
    } else {
        assert( rc_status == HIT );
        switch ( request_format ) {
        case NO_TRANSLATION: 
            // 3. region cache is swizzled (hit) and request is default => evict region from L2, invalidate mapping in region cache, ack
            // TODO: evict region from L2
            m_vm_region_cache->invalidate(mf);
            m_stats->vm_region_evict_same_change_to_default++;
            break;
        case DYNAMIC_TRANSLATION:
            // 4. region cache is swizzled (hit) and request is swizzled
            if ( *payload->get_virtualized_page() == *region_policy.get_virtualized_page() ) {
                //    (a) swizzle parameters match => nothing to do, ack
            } else {
                //    (b) swizzle parameters differ => evict from L2, update region cache, ack
                m_stats->vm_region_evict_same_change_swizzle++;
                // TODO: evict from L2
                bool evicted=false;
                vm_page_mapping_payload evicted_policy;
                m_vm_region_cache->fill(mf,gpu_sim_cycle+gpu_tot_sim_cycle,*payload,evicted,evicted_policy);
                assert( !evicted );
            }
            break;
        default: abort();
        }
    }

    mem_fetch *ack = m_mf_allocator->alloc( mf->get_sid(),
            mf->get_tpc(),
            mf->get_addr(),
            VM_POLICY_CHANGE_ACK_RC,
            new vm_page_mapping_payload( *dynamic_cast< vm_page_mapping_payload* >( mf->get_payload() ) ) );
    ack->set_type( PROTOCOL_MSG ); 
    m_L2_icnt_queue->push(ack);

    m_icnt_L2_queue->pop();
    m_request_tracker.erase(mf);
    delete mf;
}

void memory_partition_unit::region_cache_policy_check( mem_fetch *mf )
{
    if ( m_config->m_vm_region_config.enabled() ) {
        // cases:
        // 1. region cache is default (miss) and request is default => normal access
        // 2. region cache is default (miss) and request is swizzled => send vm reconfig to requesting core (should now be default)
        // 3. region cache is swizzled (hit) and request is default => send vm reconfig to requesting core (should now be swizzled)
        // 4. region cache is swizzled (hit) and request is swizzled and 
        //     (a) swizzle parameters match => "normal" vm access 
        //     (b) swizzle paramters do not match => send vm reconfig request to requesting core (update swizzle parameters)

        vm_page_mapping_payload *payload = dynamic_cast<vm_page_mapping_payload*>(mf->get_payload());

        if( payload == NULL ) {
            // assume this request is to an always non-tiled region...  (e.g., instruction memory)
            return;
        }

        virtual_page_type request_format = payload->get_page_type();
        vm_page_mapping_payload region_policy;
        enum cache_request_status rc_status = m_vm_region_cache->access(mf,gpu_sim_cycle+gpu_tot_sim_cycle,region_policy);

        if ( rc_status == MISS ) {
            // region is mapped as default (no translation) in cache 
            switch ( request_format ) {
            case NO_TRANSLATION: 
                // 1. region cache is default (miss) and request is default => normal access
                break;
            case DYNAMIC_TRANSLATION: {
                // 2. region cache is default (miss) and request is swizzled => send vm reconfig to requesting core (should now be default)
                m_stats->vm_region_default_request_swizzled++;
                mem_fetch *ack = m_mf_allocator->alloc( mf->get_sid(),
                        mf->get_tpc(),
                        mf->get_addr(),
                        VM_POLICY_CHANGE_REQ_RC,
                        new vm_page_mapping_payload() );
                ack->set_type( PROTOCOL_MSG ); 
                m_L2_icnt_queue->push(ack);
                // TODO: satisfying this request could take multiple cache lookups which we do not yet model... for now, do "normal" access
                break; }
            default: abort(); /* not supported */
            }
        } else {
            // region is mapped as swizzled in cache
            switch ( request_format ) {
            case NO_TRANSLATION: {
                // case 3. region cache is swizzled (hit) and request is default => send vm reconfig to requesting core (should now be swizzled)
                m_stats->vm_region_swizzled_request_default++;
                mem_fetch *ack = m_mf_allocator->alloc( mf->get_sid(),
                        mf->get_tpc(),
                        mf->get_addr(),
                        VM_POLICY_CHANGE_REQ_RC,
                        new vm_page_mapping_payload(region_policy));
                ack->set_type( PROTOCOL_MSG ); 
                m_L2_icnt_queue->push(ack);
                // TODO: satisfying this request could take multiple cache lookups which we do not yet model... for now, do "normal" access
                break; }
            case DYNAMIC_TRANSLATION: /* case 4 */
                if ( *payload->get_virtualized_page() == *region_policy.get_virtualized_page() ) {
                    // 4. region cache is swizzled (hit) and request is swizzled and 
                    //     (a) swizzle parameters match => "normal" vm access 
                } else {
                    // 4. region cache is swizzled (hit) and request is swizzled and 
                    //     (b) swizzle paramters do not match => send vm reconfig request to requesting core (update swizzle parameters)
                    m_stats->vm_region_swizzled_request_swizzled++;
                    mem_fetch *ack = m_mf_allocator->alloc( mf->get_sid(),
                            mf->get_tpc(),
                            mf->get_addr(),
                            VM_POLICY_CHANGE_REQ_RC,
                            new vm_page_mapping_payload(region_policy) );
                    ack->set_type( PROTOCOL_MSG ); 
                    m_L2_icnt_queue->push(ack);
                    // TODO: satisfying this request could take multiple cache lookups which we do not yet model... for now, do "normal" access
                }
                break;
            default: abort(); /* not supported */
            } 
        }
    }
}

void L2interface::push(mem_fetch *mf)
{
   // Here we need to translate this block in remapped format into a series of memory requests in linear format
   if ( mf->get_payload() && dynamic_cast< vm_page_mapping_payload* >( mf->get_payload() )->get_page_type() != VP_NON_TRANSLATED ) {
     std::list< mem_fetch* > exploded_mf = reverse_transform_mf_to_mf_list( mf );
     //printf( "L2interface - mf->get_addr()=%p explodes to %zu fetches\n", (void*)mf->get_addr(), exploded_mf.size() );
     int count = 0;
     for ( std::list< mem_fetch* >::const_iterator it = exploded_mf.begin(); it != exploded_mf.end(); ++it ) {
        //printf( "%d Reversed to %p size = %u\n", count, (void*)(*it)->get_addr(), (*it)->get_data_size() );
        (*it)->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,0/*FIXME*/);
        m_unit->m_L2_dram_queue->push(*it);
        count++;
     }
   } else {
      mf->set_status(IN_PARTITION_L2_TO_DRAM_QUEUE,0/*FIXME*/);
      m_unit->m_L2_dram_queue->push(mf);
   }
}

#define BYPASS_FETCH_EXPLOSION_AT_L2 1
std::list< mem_fetch* > L2interface::reverse_transform_mf_to_mf_list( mem_fetch* base_memfetch ) const {
#if BYPASS_FETCH_EXPLOSION_AT_L2
   std::list< mem_fetch* > result;
   result.push_back( base_memfetch );
#else
   std::list< mem_fetch* > result;
   assert( base_memfetch->get_payload() );
   const virtualized_tiled_region& region = dynamic_cast< vm_page_mapping* >( base_memfetch->get_payload() )->get_region_tile_definition();

   std::map< new_addr_type, unsigned > address_list;
   // For each byte in this mem_fetch block, find it's identifier, reverse transform the address then merge
   for ( new_addr_type current_byte = base_memfetch->get_addr(); current_byte < base_memfetch->get_addr() + base_memfetch->get_data_size(); ++current_byte ) {
      virtual_access_identifier ident = region.reverse_translate( current_byte );
      const new_addr_type original_addr = region.reverse_translate( ident );
      address_list[ original_addr ] = 1;
   }


   // For now I am not going to filter, lets just blast out requests
   unsigned contiguous_count = 1;
   std::map< new_addr_type, unsigned >::const_iterator it = address_list.begin();
   new_addr_type start_contig_addr = it->first;
   new_addr_type last_addr = it->first;
   ++it;
   vm_exploded_L2_payload* pay = new vm_exploded_L2_payload( base_memfetch );
   for ( ; it != address_list.end(); ++it ) {
      if ( last_addr == ( it->first - 1 ) ) {
         contiguous_count++;
      } else {
         pay->increment_exploded_in_flight();
         mem_fetch* next_fetch = m_unit->m_mf_allocator->alloc( base_memfetch->get_sid(), base_memfetch->get_tpc(), start_contig_addr, base_memfetch->get_access_type(), contiguous_count, base_memfetch->get_is_write(), pay );
         result.push_back( next_fetch );
         start_contig_addr = it->first;
         contiguous_count = 1;
      }
      last_addr = it->first;
   }
   // fire off the last group
   pay->increment_exploded_in_flight();
   mem_fetch* next_fetch = m_unit->m_mf_allocator->alloc( base_memfetch->get_sid(), base_memfetch->get_tpc(), start_contig_addr, base_memfetch->get_access_type(), contiguous_count, base_memfetch->get_is_write(), pay );
   result.push_back( next_fetch );

   //printf( "L2interface explodes %p into %zu accesses\n", (void*)base_memfetch->get_addr(), result.size() );
#endif
   return result;
}
