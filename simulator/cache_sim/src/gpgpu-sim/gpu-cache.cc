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

#include "gpu-cache.h"
///#include "stat-tool.h"
//#include "shader.h"
#include <assert.h>
//#include "../cuda-sim/cuda-sim.h"
//#include "gpu-sim.h"
#include "../cache_oracle.h"

extern cache_oracle* g_cache_oracle;
extern std::list< access_stream_entry >::const_iterator g_access_stream_iterator;
extern std::map< unsigned, std::list< std::list< access_stream_entry > > >
        g_warp_to_pc_bundle_to_access_list;

unsigned PROTECTED_CYCLES = 0;
unsigned PROTECTED_ACCESES = 0;

const char access_stream_entry::m_status_to_char_map[ NUM_CACHE_REQUEST_STATUS ]
                                                      = {'h','r','m','f','p'};

base_tag_array::~base_tag_array() {
}

base_tag_array::base_tag_array( const cache_config &config, int core_id, int type_id, bool enable_access_stream_dump )
: m_config(config), m_stats( this ), m_enable_access_stream_dump( enable_access_stream_dump )
{
    assert( m_config.m_write_policy == READ_ONLY );

    m_lines.resize( config.get_num_lines(), NULL );
   
    // initialize snapshot counters for visualizer
    m_prev_snapshot_access = 0;
    m_prev_snapshot_miss = 0;
    m_prev_snapshot_pending_hit = 0;
    m_core_id = core_id; 
    m_type_id = type_id;

    m_access = 0;
    m_miss = 0;
    m_pending_hit = 0;
}

void base_tag_array::print_tag_info() const{
	const char *blk_state[4] = {"INVALID",
			    "RESERVED",
			    "VALID",
			    "MODIFIED"};

	printf("Tag_array: %u lines, m_access: %u, m_miss: %u, m_pending_hit: %u\n"
		   "           m_prev_snapshot_access: %u, m_prev_snapshot_miss: %d, m_prev_snapshot_pending_hit: %d, m_core_id: %d, m_type_id: %d\n", 
           m_config.get_num_lines(),
           m_access, m_miss, m_pending_hit, m_prev_snapshot_access, m_prev_snapshot_miss, m_prev_snapshot_pending_hit, m_core_id, m_type_id);
    printf("Tag_array blocks that are not invalid ==> \n");
	for(unsigned i=0; i<m_config.get_num_lines(); i++){
        if( m_lines[i]->m_status != INVALID ) {
    		printf("idx=%4u, m_tag 0x%llx, m_block_addr: 0x%llx, m_alloc_time: %u, m_last_access_t: %u, m_fill_time: %u, cache_blk_state: %s\n",
    			i, m_lines[i]->m_tag, m_lines[i]->m_block_addr, m_lines[i]->m_alloc_time, m_lines[i]->m_last_access_time, m_lines[i]->m_fill_time,
    							blk_state[m_lines[i]->m_status]);
        }
	}
    printf("<==\n");
}

enum cache_request_status base_tag_array::probe( new_addr_type addr, unsigned &idx ) const {
    bool dummy;
    return probe( addr, idx, dummy );
}

enum cache_request_status base_tag_array::probe( new_addr_type addr, unsigned &idx, bool &all_reserved ) const {
    assert( m_config.m_write_policy == READ_ONLY );
    unsigned set_index = m_config.set_index(addr);
    new_addr_type tag = m_config.tag(addr);

    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned valid_timestamp = (unsigned)-1;
    unsigned furthest_in_future_position = 0;

    all_reserved = true;

    // check for hit or pending hit
    for (unsigned way=0; way<m_config.m_assoc; way++) {
        unsigned index = set_index*m_config.m_assoc+way;
        cache_block_t *line = m_lines[index];
        if (line->m_tag == tag) {
            if ( line->m_status == RESERVED ) {
                idx = index;
                return HIT_RESERVED;
            } else if ( line->m_status == VALID ) {
                idx = index;
                return HIT;
            } else if ( line->m_status == MODIFIED ) {
                idx = index;
                return HIT;
            } else {
                assert( line->m_status == INVALID );
            }
        }
        if (line->m_status != RESERVED) {
            all_reserved = false;
            if (line->m_status == INVALID) {
                invalid_line = index;
            } else {
                // valid line : keep track of most appropriate replacement candidate
                if ( m_config.m_replacement_policy == BELADY_OPTIMAL ) {
                    unsigned long long my_future_rr_pos
                        = g_cache_oracle->get_future_access_position( line->m_block_addr,
                                                                      g_access_stream_iterator );
                    if ( my_future_rr_pos > furthest_in_future_position ) {
                        furthest_in_future_position = my_future_rr_pos;
                        valid_line = index;
                    }
                } else if ( PROTECT_ALL_PENDING_DATA == m_config.m_replacement_policy ) {
                    bool is_to_be_protected = false;
                    for( std::map< unsigned, std::list< std::list< access_stream_entry > > >::const_iterator iter =
                            g_warp_to_pc_bundle_to_access_list.begin(); iter != g_warp_to_pc_bundle_to_access_list.end();
                            ++ iter ) {
                       if ( iter->second.size() > 0 ) {
                           for ( std::list< access_stream_entry >::const_iterator iter2 = iter->second.front().begin();
                                   iter2 != iter->second.front().end(); ++iter2 ) {
                               if ( m_config.block_addr( iter2->get_addr() ) == line->m_block_addr ) {
                                   is_to_be_protected = true;
                                   break;
                               }
                           }
                       }
                       if ( is_to_be_protected ) break;
                    }
                    if ( line->m_last_access_time < valid_timestamp && !is_to_be_protected ) {
                        valid_timestamp = line->m_last_access_time;
                        valid_line = index;
                    }
                } else if ( m_config.m_replacement_policy == LRU ) {
                    if ( line->m_last_access_time < valid_timestamp ) {
                        valid_timestamp = line->m_last_access_time;
                        valid_line = index;
                    }
                } else if ( m_config.m_replacement_policy == FIFO ) {
                    if ( line->m_alloc_time < valid_timestamp ) {
                        valid_timestamp = line->m_alloc_time;
                        valid_line = index;
                    }
                } else if ( m_config.is_rrip_policy() ) {
                    if ( rrip_cache_block::EVICT_RRPV == dynamic_cast< rrip_cache_block* >( line )->m_rrpv ) {
                        valid_line = index;
                    }
                }
            }
        }
    }
    if ( all_reserved ) {
        assert( m_config.m_alloc_policy == ON_MISS ); 
        return RESERVATION_FAIL; // miss and not enough space in cache to allocate on miss
    }

    if ( invalid_line != (unsigned)-1 ) {
        idx = invalid_line;
    } else if ( valid_line != (unsigned)-1) {
        idx = valid_line;
    } else if ( m_config.is_rrip_policy() ) {
        return RESERVATION_FAIL;
    } else abort(); // if an unreserved block exists, it is either invalid or replaceable

    return MISS;
}

enum cache_request_status base_tag_array::access( new_addr_type addr, unsigned time, unsigned &idx, const mem_fetch* mf )
{
    bool wb=false;
    cache_block_t evicted( m_config.get_line_sz() );
    enum cache_request_status result = access( addr, time,idx, wb, evicted, mf );
    //assert(!wb);
    return result;
}

enum cache_request_status base_tag_array::access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted, const mem_fetch* mf )
{
    m_access++;
    //shader_cache_access_log(m_core_id, m_type_id, 0); // log accesses to cache
    enum cache_request_status status = probe(addr,idx);

    if ( m_enable_access_stream_dump ) {
        m_stats.m_access_stream.push_back(
                access_stream_entry( addr, NUM_MEM_ACCESS_TYPE, NUM_CACHE_REQUEST_STATUS, mf ) );
    }

    switch (status) {
    case HIT_RESERVED: 
        m_pending_hit++;
    case HIT: 
        m_lines[idx]->m_last_access_time=time;
        break;
    case MISS:
        m_miss++;
        //shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        if ( m_config.m_alloc_policy == ON_MISS ) {
            if( m_lines[idx]->m_status == MODIFIED) {
                wb = true;
                evicted = *m_lines[idx];
            }
            m_lines[idx]->allocate( m_config.tag(addr), m_config.block_addr(addr), time );
        }
        break;
    case RESERVATION_FAIL:
        m_miss++;
        //shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        break;
    default:
        fprintf( stderr, "Unknown cache_request_status\n" );
        abort();
    }
    return status;
}

enum cache_request_status distill_tag_array::access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted ) {
    enum cache_request_status status = base_tag_array::access( addr, time, idx, wb, evicted, NULL );
    switch (status) {
    case HIT:
        m_lines[idx]->m_footprint.set(m_config.word_id(addr));   // Mark word as used - LDIS cache
        break;
    case MISS:
        if ( m_config.m_alloc_policy == ON_MISS ) {
            // Tayler FIXME: Temporary Hack
            if( m_lines[idx]->m_status == MODIFIED || m_lines[idx]->m_footprint.count() > 0) {
                wb = true;
                evicted = *m_lines[idx];
            }
            m_lines[idx]->m_footprint.set(m_config.word_id(addr));   // Mark word as used - LDIS cache
        }
        break;
    default:
        break; // Do Nothing
    }
    return status;
}

void base_tag_array::fill( new_addr_type addr, unsigned time )
{
    assert( m_config.m_alloc_policy == ON_FILL );
    unsigned idx;
    enum cache_request_status status = probe(addr,idx);
    assert(status==MISS); // MSHR should have prevented redundant memory request
    m_lines[idx]->allocate( m_config.tag(addr), m_config.block_addr(addr), time );
    m_lines[idx]->fill(time);
}

void base_tag_array::fill( unsigned index, unsigned time )
{
    assert( m_config.m_alloc_policy == ON_MISS );
    m_lines[index]->fill(time);
}

void base_tag_array::flush()
{
    for (unsigned i=0; i < m_config.get_num_lines(); i++) {
        m_lines[i]->m_status = INVALID;
    }
    if ( m_enable_access_stream_dump ) {
        m_stats.m_access_stream.push_back(
                access_stream_entry( access_stream_entry::CACHE_FLUSH_SIGNATURE,
                NUM_MEM_ACCESS_TYPE,
                NUM_CACHE_REQUEST_STATUS, NULL ) );
    }
}

float base_tag_array::windowed_miss_rate( bool minus_pending_hit ) const
{
    unsigned n_access    = m_access - m_prev_snapshot_access;
    unsigned n_miss      = m_miss - m_prev_snapshot_miss;
    unsigned n_pending_hit = m_pending_hit - m_prev_snapshot_pending_hit;

    if (minus_pending_hit)
        n_miss -= n_pending_hit;
    float missrate = 0.0f;
    if (n_access != 0)
        missrate = (float) n_miss / n_access;
    return missrate;
}

void base_tag_array::print_set( new_addr_type addr, FILE *stream ) const
{
    unsigned set_index = m_config.set_index(addr);
    fprintf(stream,"printing cache set for address 0x%llx\n", addr );
    for (unsigned way=0; way<m_config.m_assoc; way++) {
        unsigned index = set_index*m_config.m_assoc+way;
        assert(index < m_config.get_num_lines() );
        cache_block_t *line = m_lines[index];
        fprintf(stream,"  way %2u: ", way );
        if( line ) 
            line->print(stream);
    }
}

void base_tag_array::new_window()
{
    m_prev_snapshot_access = m_access;
    m_prev_snapshot_miss = m_miss;
    m_prev_snapshot_pending_hit = m_pending_hit;
}

void base_tag_array::print( FILE *stream, unsigned &total_access, unsigned &total_misses ) const
{
    m_config.print(stream);
    fprintf( stream, "\t\tAccess = %d, Miss = %d (%.3g), -MgHts = %d (%.3g)\n", 
             m_access, m_miss, (float) m_miss / m_access, 
             m_miss - m_pending_hit, (float) (m_miss - m_pending_hit) / m_access);
    total_misses+=m_miss;
    total_access+=m_access;
}


/*
*	Line Distillation Cache
*/
enum distill_cache_request_status distill_tag_array::probe( new_addr_type addr, unsigned &idx, unsigned &woc_idx  ) const {
    assert( m_config.m_write_policy == READ_ONLY );

    unsigned set_index = m_config.set_index(addr);

    new_addr_type tag = m_config.tag(addr);

    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned valid_timestamp = (unsigned)-1;

    bool all_reserved = true;

    unsigned num_woc_lines = m_config.m_assoc - (m_config.m_assoc * 3)/4; // 8 ways ==> 2 WOC lines
    woc_idx = (unsigned)-1;
    // Temporary assert
    assert(num_woc_lines == 2);

    // check for hit or pending hit
    unsigned way = 0;
    for (way=0; way< (m_config.m_assoc - num_woc_lines); way++) {
        unsigned index = set_index*m_config.m_assoc+way;
        assert(index < m_config.get_num_lines());
        cache_block_t *line = m_lines[index];

        if (line->m_tag == tag) {	
            if ( line->m_status == RESERVED ) {
                idx = index;
                return LOC_HIT_RESERVED;
            } else if ( line->m_status == VALID ) {
                idx = index;
                return LOC_HIT;
            } else if ( line->m_status == MODIFIED ) {
                idx = index;
                return LOC_HIT;
            } else {
                assert( line->m_status == INVALID );
            }
        }
        if (line->m_status != RESERVED) { 
            all_reserved = false;
            if (line->m_status == INVALID) {
                invalid_line = index;
            } else {
                // valid line : keep track of most appropriate replacement candidate
                if ( m_config.m_replacement_policy == LRU || 
                     m_config.m_replacement_policy == DIVERGENT_MRU ||
                     m_config.m_replacement_policy == STATIC_LOAD_PREDICTION ) {
                    if ( line->m_last_access_time < valid_timestamp ) {
                        valid_timestamp = line->m_last_access_time;
                        valid_line = index;
                    }
                } else if ( m_config.m_replacement_policy == FIFO ) {
                    if ( line->m_alloc_time < valid_timestamp ) {
                        valid_timestamp = line->m_alloc_time;
                        valid_line = index;
                    }
                }
            }
        }	
    } // No hit in the LOC --> Check the WOC
    assert(way == m_config.m_assoc - num_woc_lines);
    bool hole_miss = false;
    // Temporary assert
    for(; way < m_config.m_assoc; way++){
	unsigned index = set_index*m_config.m_assoc+way;		
	assert(index < m_config.get_num_lines());
    woc_cache_block_t *line = m_lines[index];
	for(unsigned j=0; j<(m_config.get_line_sz()/WORD_SIZE); j++){ // Check each word for a hit
		if(tag == line->WOC[j].tag && line->WOC[j].state == VALID){
			if(m_config.word_id(addr) == line->WOC[j].word_id){
				idx = index;
				return WOC_HIT;
			}else{
				woc_idx = index; // Save the WOC block where the hit occurred. 
				hole_miss = true;
			}	
		}
					
	}
    }
 
    if( all_reserved ) {
        assert( m_config.m_alloc_policy == ON_MISS ); 
        return DISTILL_RESERVATION_FAIL; // miss and not enough space in cache to allocate on miss
    }
    if ( invalid_line != (unsigned)-1 ) {
        idx = invalid_line;
    } else if ( valid_line != (unsigned)-1) {
        idx = valid_line;
    } else abort(); // if an unreserved block exists, it is either invalid or replaceable 


    if(hole_miss){ // The tag was in the WOC but the word wasn't -> idx was set to correct WOC line
	return HOLE_MISS;
    }

    return LINE_MISS;

}

enum distill_cache_request_status distill_tag_array::access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted, cache_block_t &evicted_woc_block){

    unsigned woc_idx = (unsigned)-1;
    enum distill_cache_request_status status = probe(addr,idx, woc_idx);    
    new_addr_type tag = m_config.tag(addr);
    switch (status) {
    case LOC_HIT_RESERVED: 
        // Treat like a regular hit...
    case LOC_HIT: 
        m_lines[idx]->m_last_access_time=time;
        break;
    case WOC_HIT:
	m_lines[idx]->m_last_access_time=time;
	break;
    case HOLE_MISS:
	// Invalidate all words with  this tag -> write out if dirty. 
	assert(woc_idx != (unsigned)-1);
	evicted_woc_block = *m_lines[woc_idx]; // This is the evicted WOC block, it contains words matching tag -> write back modified words
	for(unsigned i=0; i<words_per_block; i++){
		if((*m_lines[woc_idx]).WOC[i].tag == tag){
			(*m_lines[woc_idx]).WOC[i].state = INVALID;
			(*m_lines[woc_idx]).WOC[i].tag = 0;
		}
	}
    case LINE_MISS:
	// Not in LOC, Not in WOC
        if ( m_config.m_alloc_policy == ON_MISS ) {
            if( m_lines[idx]->m_status == MODIFIED ) {
                wb = true;	
                evicted = *m_lines[idx];
            }
  	    // If number of used blocks is below threshold, Save used blocks in WOC
	    unsigned footprint_count = m_lines[idx]->m_footprint.count();
	    if(footprint_count <= (words_per_block/2)  && footprint_count != 0){	// Need to implement the Threshold-Based Distillation here Currently just half of line
	    	save_used_words(m_lines[idx]->m_block_addr, idx);
	    }
            m_lines[idx]->allocate( m_config.tag(addr), m_config.block_addr(addr), m_config.word_id(addr), time ); // Overwrite line in LOC
	    m_lines[idx]->m_footprint.reset();
        }
        break;
    case DISTILL_RESERVATION_FAIL:
       	assert(0);
        break;
    }
    return status;
}

void distill_tag_array::fill( new_addr_type addr, unsigned time ){
	assert( m_config.m_alloc_policy == ON_FILL );
	unsigned idx, woc_idx;
	enum distill_cache_request_status status = probe(addr,idx, woc_idx);
        unsigned footprint_count = m_lines[idx]->m_footprint.count();
	assert(status==LINE_MISS || status == HOLE_MISS); // MSHR should have prevented redundant memory request
        if(footprint_count <= (words_per_block/2) && footprint_count != 0){	// Need to implement the Threshold-Based Distillation here - Currently just half of line
	  	save_used_words(m_lines[idx]->m_block_addr, idx);
	}
	m_lines[idx]->allocate( m_config.tag(addr), m_config.block_addr(addr), m_config.word_id(addr), time );
	m_lines[idx]->fill(time);
	m_lines[idx]->m_footprint.reset();

	return;
}
void distill_tag_array::fill( unsigned idx, unsigned time ){
        assert( m_config.m_alloc_policy == ON_MISS );
        m_lines[idx]->fill(time);
	m_lines[idx]->m_footprint.reset();

	return;
}


void distill_tag_array::save_used_words(new_addr_type addr, unsigned idx){
	unsigned footprint_count = m_lines[idx]->m_footprint.count();

	unsigned set_index = m_config.set_index(addr);

	new_addr_type tag = m_config.tag(addr);		
	unsigned num_woc_lines = m_config.m_assoc - (m_config.m_assoc * 3)/4; // 8 ways ==> 2 WOC lines
	assert(num_woc_lines == 2);
	unsigned way = m_config.m_assoc - num_woc_lines;
	unsigned index = set_index*m_config.m_assoc+way;
	
	// select which WOC line based on your address
	assert(addr % num_woc_lines <= 1);
	way += addr % num_woc_lines;
	
	unsigned inc = 1;	// Must start on multiple of number of words used (i.e. if 2 words used, must start on 0, 2, 4, 6)
	if(footprint_count == 2 || footprint_count == 3)
		inc = 2;
	else if(footprint_count == 4)
		inc = 4;
	else if(footprint_count > 4)
		inc = 8;

	unsigned stride_length = 0;
	int stride_start = -1;

	unsigned word_index = 0;

	for(unsigned j=0; j<words_per_block; j+=inc){ // Look for stride to fit words
		if(m_lines[index]->WOC[j].state == INVALID){ // Free word space
			if(stride_start == -1) stride_start = j;
			stride_length++;				
			if(footprint_count == stride_length){ // Found an open spot
				for(unsigned k=stride_start; k<footprint_count+stride_start; k++){
					m_lines[index]->WOC[k].tag = tag;
					for(;word_index < words_per_block; word_index++){
						if(m_lines[idx]->m_footprint[word_index])
							break;
					}
						
					m_lines[index]->WOC[k].word_id = word_index; //m_config.word_id(addr);
					m_lines[index]->WOC[k].state = VALID;
					if(k == (unsigned)stride_start)
						m_lines[index]->WOC[k].head_bit = true;
					word_index++;
				}
				return;
			}	
		}else{
			stride_start = -1;
		}

	}
	// No free spaces, choose random location to evict
	word_index = 0;
	stride_start = addr%(words_per_block/inc) * inc; // Basically deterministic-sudo randomly choosing which alligned index to overwrite
	for(unsigned j=stride_start; j<stride_start+footprint_count; j++){
		if(m_lines[index]->WOC[j].state == MODIFIED){
			// TODO: Write back word
		}
		m_lines[index]->WOC[j].tag = tag;
		for(;word_index < words_per_block; word_index++){
			if(m_lines[idx]->m_footprint[word_index])
				break;
		}
		m_lines[index]->WOC[j].word_id = word_index;// m_config.word_id(addr);
		m_lines[index]->WOC[j].state = VALID;
		if(j == (unsigned)stride_start)
			m_lines[index]->WOC[j].head_bit = true;
		word_index++;
	}

return;

}


enum sector_cache_request_status sector_tag_array::probe_sector( new_addr_type addr, unsigned &idx, bool &all_reserved ) const{

    assert( m_config.m_write_policy == READ_ONLY );
    unsigned set_index = m_config.set_index(addr);
    new_addr_type tag = m_config.tag(addr);
	unsigned word_id = m_config.word_id(addr);

    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned valid_timestamp = (unsigned)-1;

    all_reserved = true;

    // check for hit or pending hit
    for (unsigned way=0; way<m_config.m_assoc; way++) {
        unsigned index = set_index*m_config.m_assoc+way;
        sectored_cache_block_t *line = m_lines[index];
        if (line->m_tag == tag) {
            if ( line->word_state[word_id] == RESERVED ) {
                idx = index;
                return SECTOR_HIT_RESERVED;
            } else if ( line->word_state[word_id] == VALID ) {
                idx = index;
                return SECTOR_HIT;
            } else if ( line->word_state[word_id] == MODIFIED ) {
                idx = index;
                return SECTOR_HIT;
            } else if(line->m_status == VALID || line->m_status == RESERVED){ // Line is in (or will be) in the cache but the requested word is not
				idx = index;
				return SECTOR_WORD_MISS;
			}else {
                assert( line->m_status == INVALID );
            }
        }

        if (line->m_status != RESERVED) {
            all_reserved = false;
            if (line->m_status == INVALID) {
                invalid_line = index;
            } else {
                // valid line : keep track of most appropriate replacement candidate
                if ( m_config.m_replacement_policy == LRU ) {
                    if ( line->m_last_access_time < valid_timestamp ) {
                        valid_timestamp = line->m_last_access_time;
                        valid_line = index;
                    }
                } else if ( m_config.m_replacement_policy == FIFO ) {
                    if ( line->m_alloc_time < valid_timestamp ) {
                        valid_timestamp = line->m_alloc_time;
                        valid_line = index;
                    }
                } else if ( m_config.is_rrip_policy() ) {
                    if ( rrip_cache_block::EVICT_RRPV == dynamic_cast< rrip_cache_block* >( line )->m_rrpv ) {
                        valid_line = index;
                    }
                }
            }
        }
    }
    if ( all_reserved ) {
        assert( m_config.m_alloc_policy == ON_MISS ); 
        return SECTOR_RESERVATION_FAIL; // miss and not enough space in cache to allocate on miss
    }

    if ( invalid_line != (unsigned)-1 ) {
        idx = invalid_line;
    } else if ( valid_line != (unsigned)-1) {
        idx = valid_line;
    } else if ( m_config.is_rrip_policy() ) {
        return SECTOR_RESERVATION_FAIL;
    } else abort(); // if an unreserved block exists, it is either invalid or replaceable

    return SECTOR_MISS; // Line is not in the cache

}
enum sector_cache_request_status sector_tag_array::access_sector( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted){
    m_access++;
    //shader_cache_access_log(m_core_id, m_type_id, 0); // log accesses to cache
	bool all_reserved;
    enum sector_cache_request_status status = probe_sector(addr,idx, all_reserved 	);

    switch (status) {
    case SECTOR_HIT_RESERVED: // Word has been reserved
        m_pending_hit++;
    case SECTOR_HIT: // Hit on word
        m_lines[idx]->m_last_access_time=time;
        break;
    case SECTOR_WORD_MISS:	// Line is in the cache, word is not
        m_miss++;
        //shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        if ( m_config.m_alloc_policy == ON_MISS ) {
            m_lines[idx]->allocate( m_config.tag(addr), m_config.block_addr(addr), m_config.word_id(addr), time );
        }
        break;
	case SECTOR_MISS:	// Line is not in the cache
        if ( m_config.m_alloc_policy == ON_MISS ) {
			if( m_lines[idx]->m_status == MODIFIED) { // Write back modified words
				wb = true;
				evicted = *m_lines[idx];
			}
			m_lines[idx]->allocate( m_config.tag(addr), m_config.block_addr(addr), m_config.word_id(addr), time );
		}
		break;
    case SECTOR_RESERVATION_FAIL:
        m_miss++;
        //shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        break;
    default:
        fprintf( stderr, "Unknown cache_request_status\n" );
        abort();
    }
    return status;
}


enum cache_request_status sector_cache::access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ){
	// Tayler - Very little modifications from the baseline access function.. I'll integrate it once debugged

	// addr needs to be actual address of 
	bool wr = mf->get_is_write();
	enum mem_access_type type = mf->get_access_type();
	bool evict = (type == GLOBAL_ACC_W); // evict a line that hits on global memory write

	new_addr_type block_addr = m_config.block_addr(addr);
	//unsigned word_id = m_config.word_id(addr);
	m_stats.access_event( block_addr );

	unsigned cache_index = (unsigned)-1;

	bool all_reserved = true;
	enum sector_cache_request_status status = m_tag_array->probe_sector( block_addr, cache_index, all_reserved );

	if(status == SECTOR_HIT){
		if(evict){
			if ( m_miss_queue.size() >= m_config.m_miss_queue_size )
				return RESERVATION_FAIL; // cannot handle request this cycle

			// generate a write through
			sectored_cache_block_t &block = m_tag_array->get_block(cache_index);
			assert( block.m_status != MODIFIED ); // fails if block was allocated by a ld.local and now accessed by st.global

			m_miss_queue.push_back(mf);
			mf->set_status(m_miss_queue_status,time);
			events.push_back(WRITE_REQUEST_SENT);

			// invalidate block and sectors
			block.m_status = INVALID;
			for(unsigned i=0; i<NUM_SECTORED; i++)
				block.word_state[i] = INVALID;

		}else{
			sectored_cache_block_t evicted( m_config.get_line_sz() );
			bool wb= false;
			status = m_tag_array->access_sector(addr,time,cache_index, wb, evicted); 
			//m_stats.hit_event( addr, (cache_block_t &)evicted, cache_index, mf, status );
			if ( wr ) {
				assert( type == LOCAL_ACC_W ||  type == L1_WRBK_ACC );
				// treated as write back...
				sectored_cache_block_t &block = m_tag_array->get_block(cache_index);

				block.m_status = MODIFIED;	// Mark block and sector as MODIFIED
				block.word_state[m_config.word_id(addr)] = MODIFIED;
			}
		}
	}else if(status != SECTOR_RESERVATION_FAIL){
		if(wr){
			if ( m_miss_queue.size() >= m_config.m_miss_queue_size )
				return RESERVATION_FAIL; // cannot handle request this cycle

			// on miss, generate write through (no write buffering -- too many threads for that)
			mf->set_footprint(m_config.word_id(addr), 1); // Mark this word as being used
			m_miss_queue.push_back(mf);
			mf->set_status(m_miss_queue_status,time);
			events.push_back(WRITE_REQUEST_SENT);
		 	return MISS;
		}else{
			if ( (m_miss_queue.size()+1) >= m_config.m_miss_queue_size )
				return RESERVATION_FAIL; // cannot handle request this cycle (might need to generate two requests)

			bool do_miss = false;
			bool wb = false;
			cache_block_t evicted( m_config.get_line_sz() );

			bool mshr_hit = m_mshrs.probe(block_addr);
			bool mshr_avail = !m_mshrs.full(block_addr);
			mf->set_footprint(m_config.word_id(addr), 1); // If going to merge with an existing miss, need to update the footprint in the MSHR

			if ( mshr_hit && mshr_avail ) {
				status = m_tag_array->access_sector(addr,time,cache_index,wb,evicted);
				//m_stats.miss_event( addr, evicted, cache_index, mf, status );
				m_mshrs.add(block_addr,mf);
				do_miss = true;
			} else if ( !mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size) ) {
				status = m_tag_array->access_sector(addr,time,cache_index,wb,evicted);
				//m_stats.miss_event( addr, evicted, cache_index, mf, status );
				m_mshrs.add(block_addr,mf);
				m_extra_mf_fields[mf] = extra_mf_fields(block_addr,cache_index, mf->get_data_size());
	
				mf->set_data_size( m_config.get_line_sz() );
				m_miss_queue.push_back(mf);
				mf->set_status(m_miss_queue_status,time);
				events.push_back(READ_REQUEST_SENT);
				do_miss = true;
			}
			if( wb ) {
				assert(do_miss);
				mem_fetch *wb_mf = m_memfetch_creator->alloc(evicted.m_block_addr,L1_WRBK_ACC,m_config.get_line_sz(),true);
				events.push_back(WRITE_BACK_REQUEST_SENT);
				m_miss_queue.push_back(wb_mf);
				wb_mf->set_status(m_miss_queue_status,time);
			}
			if( do_miss ) {
				return MISS;
			}

		}

	}

	return RESERVATION_FAIL;
}

void sector_cache::fill( mem_fetch *mf, unsigned time ){

}

void rrip_tag_array::handle_hit( new_addr_type block_addr ) {
    unsigned set_index = m_config.set_index( block_addr );
    new_addr_type tag = m_config.tag( block_addr );
    for (unsigned way=0; way<m_config.m_assoc; way++) {
        unsigned index = set_index*m_config.m_assoc+way;
        rrip_cache_block *line = m_lines[index];
        if ( line->m_tag == tag ) {
            if ( HP_RRIP_HIT_POLICY == m_config.mrrip_hit_update_pol ) {
                line->m_rrpv = 0;
            } else if ( FP_RRIP_HIT_POLICY == m_config.mrrip_hit_update_pol ) {
                if ( line->m_rrpv > 0 ) {
                    --line->m_rrpv;
                }
            }
            return;
        }
    }
    abort();
}

void rrip_tag_array::increase_rrip( new_addr_type block_addr ) {
    unsigned set_index = m_config.set_index( block_addr );
    for ( unsigned way = 0; way<m_config.m_assoc; way++ ) {
        unsigned index = set_index * m_config.m_assoc + way;
        if ( m_lines[index]->m_rrpv < ( 1 << rrip_cache_block::M ) - 1  ) {
            ++m_lines[index]->m_rrpv;
        }
        assert( m_lines[index]->m_rrpv < ( 1 << rrip_cache_block::M ) );
    }
}

void priority_tag_array::protect_line( unsigned cache_index, unsigned warp_id ) {
    priority_cache_block& block = dynamic_cast< priority_cache_block& >( get_block( cache_index ) );
    if ( get_protection_type() == CYCLES_PROTECTION_TYPE && 0 == block.m_cycles_protected ) {
        ++m_num_protected_lines;
    } else if ( get_protection_type() == ACCESSES_PROTECTION_TYPE && 0 == block.m_accesses_protected ) {
        ++m_num_protected_lines;
    }
    if ( block.m_warps_protecting_line.size() > 0 ) {
        block.m_warps_protecting_line.clear();
    }
    block.m_warps_protecting_line[ warp_id ] = 1;
    m_lines_protected_per_warp[ warp_id ].second++;
    if ( get_protection_type() == CYCLES_PROTECTION_TYPE ) {
        block.m_cycles_protected = PROTECTED_CYCLES;
    } else if ( get_protection_type() == ACCESSES_PROTECTION_TYPE ){
        block.m_accesses_protected = PROTECTED_ACCESES;
    } else {
        abort();
    }

    assert( m_num_protected_lines <= m_lines.size() );
}

bool was_write_sent( const std::list<cache_event> &events )
{
    for( std::list<cache_event>::const_iterator e=events.begin(); e!=events.end(); e++ ) {
        if( *e == WRITE_REQUEST_SENT ) 
            return true;
    }
    return false;
}

bool was_read_sent( const std::list<cache_event> &events )
{
    for( std::list<cache_event>::const_iterator e=events.begin(); e!=events.end(); e++ ) {
        if( *e == READ_REQUEST_SENT ) 
            return true;
    }
    return false;
}

enum cache_request_status evict_on_write_cache::access( new_addr_type addr,
                                                        mem_fetch *mf,
                                                        unsigned time,
                                                        std::list<cache_event> &events ) {
     bool wr = mf->get_is_write();
     enum mem_access_type type = mf->get_access_type();
     bool evict = (type == GLOBAL_ACC_W); // evict a line that hits on global memory write

     new_addr_type block_addr = m_config.block_addr(addr);
     m_stats.access_event( block_addr );

     unsigned cache_index = (unsigned)-1;
     
     bool all_reserved = true;
     enum cache_request_status status = m_tag_array->probe( block_addr, cache_index, all_reserved );
     if ( m_config.is_rrip_policy() && HIT == status ) {
        dynamic_cast< rrip_tag_array* >( m_tag_array )->handle_hit( block_addr );
     }

     while ( m_config.is_rrip_policy() && !all_reserved && RESERVATION_FAIL == status ) {
         dynamic_cast< rrip_tag_array* >( m_tag_array )->increase_rrip( block_addr );
         status = m_tag_array->probe( block_addr, cache_index, all_reserved );
     }

     if ( status == HIT ) {
         if ( evict ) {
             if ( m_miss_queue.size() >= m_config.m_miss_queue_size )
                 return RESERVATION_FAIL; // cannot handle request this cycle

             // generate a write through
             cache_block_t &block = m_tag_array->get_block(cache_index);
             assert( block.m_status != MODIFIED ); // fails if block was allocated by a ld.local and now accessed by st.global

             m_miss_queue.push_back(mf);
             mf->set_status(m_miss_queue_status,time);
             events.push_back(WRITE_REQUEST_SENT);

             // invalidate block
             block.m_status = INVALID;
             if ( m_config.enable_cache_access_dump ) {
                 m_tag_array->get_stats().m_access_stream.push_back(
                         access_stream_entry( addr, type, status, mf ) );
             }
         } else {
             cache_block_t evicted( m_config.get_line_sz() );
             bool wb= false;
             //status = m_tag_array->access(block_addr,time,cache_index, wb, evicted); // update LRU state
             status = m_tag_array->access(addr,time,cache_index, wb, evicted, mf); // TODO: Tayler - Needed actual address for word access... not just block address
             m_tag_array->get_stats().m_access_stream.back().set_access_type( type );
             m_tag_array->get_stats().m_access_stream.back().set_request_status( status );

             m_stats.hit_event( addr, evicted, cache_index, mf, status );

             if ( wr ) {
                 assert( type == LOCAL_ACC_W || /*l2 only*/ type == L1_WRBK_ACC );
                 // treated as write back...
                 cache_block_t &block = m_tag_array->get_block(cache_index);
                 block.m_status = MODIFIED;
             }
         }
         return HIT;
     } else if ( status != RESERVATION_FAIL ) {
         if ( wr ) {
             if ( m_miss_queue.size() >= m_config.m_miss_queue_size )
                 return RESERVATION_FAIL; // cannot handle request this cycle
             // on miss, generate write through (no write buffering -- too many threads for that)
             mf->set_footprint(m_config.word_id(addr), 1);
             m_miss_queue.push_back(mf);
             mf->set_status(m_miss_queue_status,time);
             events.push_back(WRITE_REQUEST_SENT);
             return MISS;
         } else {
             if ( (m_miss_queue.size()+1) >= m_config.m_miss_queue_size )
                 return RESERVATION_FAIL; // cannot handle request this cycle (might need to generate two requests)

             bool do_miss = false;
             bool wb = false;
             cache_block_t evicted( m_config.get_line_sz() );

             bool mshr_hit = m_mshrs.probe(block_addr);
             bool mshr_avail = !m_mshrs.full(block_addr);
             mf->set_footprint(m_config.word_id(addr), 1);
             if ( mshr_hit && mshr_avail ) {
                 status = m_tag_array->access( addr,time, cache_index, wb, evicted, mf );
                 m_tag_array->get_stats().m_access_stream.back().set_access_type( type );
                 m_tag_array->get_stats().m_access_stream.back().set_request_status( status );
                 m_stats.miss_event( addr, evicted, cache_index, mf, status );
                 m_mshrs.add(block_addr,mf);
                 do_miss = true;
             } else if ( !mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size) ) {
                 status = m_tag_array->access( addr,time, cache_index, wb, evicted, mf );
                 m_tag_array->get_stats().m_access_stream.back().set_access_type( type );
                 m_tag_array->get_stats().m_access_stream.back().set_request_status( status );
                 m_stats.miss_event( addr, evicted, cache_index, mf, status );
                 m_mshrs.add(block_addr,mf);
                 m_extra_mf_fields[mf] = extra_mf_fields(block_addr,cache_index, mf->get_data_size());

                 mf->set_data_size( m_config.get_line_sz() );
                 m_miss_queue.push_back(mf);
                 mf->set_status(m_miss_queue_status,time);
                 events.push_back(READ_REQUEST_SENT);
                 do_miss = true;
             }
             if( wb ) {
                 assert(do_miss);
                 mem_fetch *wb = m_memfetch_creator->alloc(evicted.m_block_addr,L1_WRBK_ACC,m_config.get_line_sz(),true);
                 events.push_back(WRITE_BACK_REQUEST_SENT);
                 m_miss_queue.push_back(wb);
                 wb->set_status(m_miss_queue_status,time);
             }
             if( do_miss ) {
                return MISS;
             }

         }
     }

     return RESERVATION_FAIL;
}

typedef std::pair< new_addr_type, unsigned > mypair;

bool cmp_seconds(const mypair &lhs, const mypair &rhs) {
    return lhs.second > rhs.second;
}

// TODO make this a runtime option
#define USE_SET_INDEX_HASH 1

unsigned cache_config::set_index( new_addr_type addr ) const
{
   unsigned set;
#if USE_SET_INDEX_HASH
   set = set_index_hashed( addr );
#else
   set = (addr >> m_line_sz_log2) & (m_nset-1);
#endif
   assert( set < m_nset );
   return set;
}

unsigned cache_config::set_index_hashed( new_addr_type addr ) const{
   new_addr_type set = (addr >> m_line_sz_log2) & (m_nset - 1);
   new_addr_type tag = (addr >> (m_line_sz_log2+m_nset_log2));
   tag *= ODD_HASH_MULTIPLE;

   set = (set + tag) & (m_nset - 1);
   return set;
}

