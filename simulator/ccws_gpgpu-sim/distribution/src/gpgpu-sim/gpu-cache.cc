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
#include "stat-tool.h"
#include "shader.h"
#include <assert.h>
#include "../cuda-sim/cuda-sim.h"
#include "gpu-sim.h"

#define DO_EVERY(X,Y) {static unsigned counter = 0;if (counter == X ) {counter = 0;Y;} else{++counter;}}

unsigned PROTECTED_CYCLES = 0;
unsigned PROTECTED_ACCESES = 0;
unsigned L1_HIT_LATENCY = 20;
unsigned AMMOUNT_ADDED_BY_THREAD = 128;
int VC_PER_WARP = 0;

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
    shader_cache_access_log(m_core_id, m_type_id, 0); // log accesses to cache
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
        shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
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
        shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        break;
    default:
        fprintf( stderr, "Unknown cache_request_status\n" );
        abort();
    }
    return status;
}

void tlb_config::init( const char* config_str /*= NULL*/ ) {
    char translation_str[255];
    const char* first_parm = strchr( m_config_string, ',');
    size_t size_of_translation_str =  (size_t)first_parm - (size_t)m_config_string;
    assert( size_of_translation_str < sizeof( translation_str ) );
    memcpy( translation_str, m_config_string, size_of_translation_str );
    translation_str[ size_of_translation_str ] = '\0';
    if ( strncmp( "none", translation_str, sizeof( translation_str ) ) == 0 ) {
        m_translation_config = NO_TRANSLATION;
    } else if ( strncmp( "static", translation_str, sizeof( translation_str ) ) == 0 ) {
        m_translation_config = STATIC_TRANSLATION;
    } else if ( strncmp( "dynamic", translation_str, sizeof( translation_str ) ) == 0 ) {
        m_translation_config = DYNAMIC_TRANSLATION;
    } else {
        printf( "Unknown TLB Config\n" );
        assert( 0 );
    }

    const char* start_of_cache_config = strchr( ++first_parm, ',');
    size_of_translation_str =  (size_t)start_of_cache_config - (size_t)first_parm;
    assert( size_of_translation_str < sizeof( translation_str ) );
    memcpy( translation_str, first_parm, size_of_translation_str );
    translation_str[ size_of_translation_str ] = '\0';
    if ( strncmp( "none", translation_str, sizeof( translation_str ) ) == 0 ) {
        m_virtual_page_type = VP_NON_TRANSLATED;
    } else if ( strncmp( "tiled", translation_str, sizeof( translation_str ) ) == 0 ) {
        m_virtual_page_type = VP_TILED;
    } else if ( strncmp( "hot_aligned", translation_str, sizeof( translation_str ) ) == 0 ) {
        m_virtual_page_type = VP_HOT_DATA_ALIGNED;
    } else if ( strncmp( "hot_unaligned", translation_str, sizeof( translation_str ) ) == 0 ) {
        m_virtual_page_type = VP_HOT_DATA_UNALIGNED;
    } else {
        printf( "Unknown page type\n" );
        assert( 0 );
    }

    g_translation_config = m_translation_config;
    g_translated_page_type = m_virtual_page_type;
    start_of_cache_config++;
    cache_config::init( start_of_cache_config );
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

float base_tag_array::windowed_miss_rate( ) const
{
    unsigned n_access    = m_access - m_prev_snapshot_access;
    unsigned n_miss      = m_miss - m_prev_snapshot_miss;
    // unsigned n_pending_hit = m_pending_hit - m_prev_snapshot_pending_hit;

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
    fprintf( stream, "\t\tAccess = %d, Miss = %d (%.3g), PendingHit = %d (%.3g)\n", 
             m_access, m_miss, (float) m_miss / m_access, 
             m_pending_hit, (float) m_pending_hit / m_access);
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
	for(unsigned j=0; j<(m_config.get_line_sz()/WOC_WORD_SIZE); j++){ // Check each word for a hit
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


enum sector_cache_request_status sector_tag_array::probe( new_addr_type addr, unsigned &idx, bool &all_reserved, std::bitset<NUM_SECTORED> mask) const{
	// Mask contains bitmask of requested sectors in this cache line

    assert( m_config.m_write_policy == READ_ONLY );
    unsigned set_index = m_config.set_index(addr);
    new_addr_type tag = m_config.tag(addr);

//	unsigned word_id = m_config.word_id(addr);

    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned valid_timestamp = (unsigned)-1;

    all_reserved = true;

    // check for hit or pending hit
    for (unsigned way=0; way<m_config.m_assoc; way++) {
        unsigned index = set_index*m_config.m_assoc+way;
        sectored_cache_block_t *line = m_lines[index];
        if (line->m_tag == tag) {
			bool all_sector_reserved = true;
			bool all_sector_valid = true;
			assert(mask.any()); // Make sure something is being requested

			// Check all requested sectors
			for(unsigned sector=0; sector<NUM_SECTORED; sector++){
				if(!mask.test(sector))
					continue;
				if(line->word_state[sector] != RESERVED)
					all_sector_reserved = false;
				if(line->word_state[sector] != VALID || line->word_state[sector] != MODIFIED)
					all_sector_valid = false;
			}
			if(all_sector_reserved){
				idx = index;
				return SECTOR_HIT_RESERVED;
			}else if(all_sector_valid){
				idx = index;
				return SECTOR_HIT;
			}else if(line->m_status == VALID || line->m_status == RESERVED){ // Line is in (or will be) in the cache but the requested word is not
				idx = index;
				return SECTOR_WORD_MISS;
			}else{
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
enum sector_cache_request_status sector_tag_array::access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted,
		std::bitset<NUM_SECTORED> mask){
    m_access++;
    shader_cache_access_log(m_core_id, m_type_id, 0); // log accesses to cache
	bool all_reserved;
    enum sector_cache_request_status status = probe(addr,idx, all_reserved, mask);

    switch (status) {
    case SECTOR_HIT_RESERVED: // Word has been reserved
        m_pending_hit++;
    case SECTOR_HIT: // Hit on word
        m_lines[idx]->m_last_access_time=time;
        break;
    case SECTOR_WORD_MISS:	// Line is in the cache, word is not
        m_miss++;
        shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
        if ( m_config.m_alloc_policy == ON_MISS ) {
			for(unsigned sector=0; sector<NUM_SECTORED; sector++)            
				m_lines[idx]->allocate( m_config.tag(addr), m_config.block_addr(addr), sector, time );
        }
        break;
	case SECTOR_MISS:	// Line is not in the cache
        if ( m_config.m_alloc_policy == ON_MISS ) {
			if( m_lines[idx]->m_status == MODIFIED) { // Write back modified words
				wb = true;
				evicted = *m_lines[idx];
			}
			for(unsigned sector=0; sector<NUM_SECTORED; sector++)
				m_lines[idx]->allocate( m_config.tag(addr), m_config.block_addr(addr), sector, time );
		}
		break;
    case SECTOR_RESERVATION_FAIL:
        m_miss++;
        shader_cache_access_log(m_core_id, m_type_id, 1); // log cache misses
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

	std::bitset<NUM_SECTORED> sector_mask = mf->get_sector_mask();

	bool all_reserved = true;
	enum sector_cache_request_status status = m_tag_array->probe( block_addr, cache_index, all_reserved, sector_mask );



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
			status = m_tag_array->access(addr,time,cache_index, wb, evicted, sector_mask); 
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
				status = m_tag_array->access(addr,time,cache_index,wb,evicted, sector_mask);
				//m_stats.miss_event( addr, evicted, cache_index, mf, status );
				m_mshrs.add(block_addr,mf);
				do_miss = true;
			} else if ( !mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size) ) {
				status = m_tag_array->access(addr,time,cache_index,wb,evicted, sector_mask);
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

bool was_vc_hit( const std::list<cache_event> &events )
{
    for( std::list<cache_event>::const_iterator e=events.begin(); e!=events.end(); e++ ) {
        if( *e == VC_HIT ) 
            return true;
    }
    return false;
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

enum cache_request_status mem_div_cache::access( new_addr_type addr, unsigned time, bool is_divergent, bool is_load_useful)
{
    unsigned index=0;
    bool wb=false;
    cache_block_t evicted( m_config.get_line_sz() );
    enum cache_request_status status = base_tag_array::access( addr, time, index, wb, evicted, NULL );
    if ( MISS == status ) {
        fill(index,time);

        if( ((m_config.get_replacement_policy() == DIVERGENT_MRU) && !is_divergent)
              || ((m_config.get_replacement_policy() == STATIC_LOAD_PREDICTION) && !is_load_useful) ) {
            // treat blocks accesses that are labelled not divergent as streaming and put into LRU position

            // find oldest LRU block's timestamp
            unsigned set_index = m_config.set_index(addr);

            unsigned oldest_timestamp=(unsigned)-1;
            unsigned oldest_index=(unsigned)-1;
            unsigned prior_oldest_timestamp=(unsigned)-1;
            unsigned prior_oldest_index=(unsigned)-1;
            for (unsigned way=0; way<m_config.m_assoc; way++) {
                unsigned idx = set_index*m_config.m_assoc+way;
                cache_block_t *line = m_lines[idx];
                if( line->m_status == VALID && line->m_last_access_time < oldest_timestamp ) {
                    prior_oldest_timestamp = oldest_timestamp;
                    prior_oldest_index = oldest_index;
                    oldest_timestamp = line->m_last_access_time;
                    oldest_index = idx;
                }
            }
            assert( oldest_timestamp != (unsigned)-1 ); // we just did a fill so at least one block is valid
            assert( oldest_timestamp != 0 ); // similarly, valid block should not have timestamp of zero

            if( prior_oldest_index != (unsigned)-1 ) {
                // if LRU block is not same as block we just inserted
                cache_block_t &block = get_block( index );
#ifdef DEBUG_DIV_CACHE
                printf("\n LRU = %u, MRU = %u ", block.m_last_access_time, oldest_timestamp-1 );
#endif
                block.m_last_access_time = oldest_timestamp-1; // LRU position
#ifdef DEBUG_DIV_CACHE
                printf("+");
#endif
            } else {
#ifdef DEBUG_DIV_CACHE
                printf("\n index = %u (set_index=%u)", index, set_index );
                printf("-");
#endif
            }
        } else {
#ifdef DEBUG_DIV_CACHE
            printf(".");
#endif
        }
    } else {
#ifdef DEBUG_DIV_CACHE
        printf("h");
#endif
    }
    return status;
}

bool mem_div_cache::is_this_load_useful( new_addr_type addr, shader_core_stats& shader_stats )
{
   const mem_div_load_counters& counters = shader_stats.all_warps_pc_to_load_counters_map[ addr ];
   unsigned long long total_useful_accesses = counters.num_times_all_bytes_used_or_eventually_used
   + counters.num_times_gte_half_bytes_used_or_eventually_used
   + counters.num_times_gte_3_quarters_bytes_used_or_eventually_used
   + counters.num_times_gte_1_quarter_used_or_eventually_used;

   return (float)total_useful_accesses / (float) counters.num_accesses > 0.4f || counters.num_accesses < 5;
}

#define ENABLE_POLICY_MAN_PRINTF 0
size_t vm_policy_manager::gcd( size_t a, size_t b ) const {
   while( 1 )
   {
      a = a % b;
      if ( a == 0 )
         return b;

      b = b % a;

      if ( b == 0 )
         return a;
   }
}

extern size_t g_hack_start_of_static_space;
extern size_t g_hack_end_of_static_space;
#define OBJECT_SIZE_MIN 128
size_t vm_policy_manager::determine_gcd_obj_size( const warp_inst_t& instr ) const {
   // Go through all the loaded addresses of this instruction and determine the gcd of the pairwise differences
   std::list< size_t > difference_list;
   new_addr_type i_addr;
   for ( unsigned i = 0; i < instr.warp_size(); ++i ) {
      if ( instr.active( i ) ) {
         i_addr = instr.get_addr( i );
         for ( unsigned j = 0; j < instr.warp_size(); ++j ) {
               if ( i_addr != instr.get_addr( j ) && instr.active( j ) ) {
                  const size_t diff = abs( i_addr - instr.get_addr( j ) );
                  difference_list.push_back( diff );
               }
         }
         break;
      }
   }
   std::map< size_t, unsigned > gcd_occurrance_map;
   std::list< size_t >::iterator it = difference_list.begin();
   const size_t first_diff = *it; ++it;
   for ( ; it != difference_list.end() ; ++it) {
      const size_t current_gcd = gcd( *it, first_diff );
      gcd_occurrance_map[ current_gcd ]++;
   }

   std::map< size_t, unsigned >::const_iterator it2 = gcd_occurrance_map.begin();
   size_t best_gcd = INVALID_OBJECT_SIZE;
   size_t smallest_gcd = INVALID_OBJECT_SIZE;
   unsigned largest_occurrance = 0;
   unsigned tot_occurrances = 0;
   for ( ; it2 != gcd_occurrance_map.end(); ++it2 ) {
      if ( it2->second > largest_occurrance ) {
         largest_occurrance = it2->second;
         best_gcd = it2->first;
      }
      if ( smallest_gcd > it2->first ) {
         smallest_gcd = it2->first;
      }
      tot_occurrances += it2->second;
   }
   it2 = gcd_occurrance_map.begin();
   bool are_all_a_multiple_of_the_smallest = true;
   for ( ; it2 != gcd_occurrance_map.end(); ++it2 ) {
      if ( smallest_gcd == INVALID_OBJECT_SIZE ) {
         are_all_a_multiple_of_the_smallest = false;
         break;
      } else if ( it2->first % smallest_gcd != 0 ) {
         are_all_a_multiple_of_the_smallest = false;
         break;
      }
   }
   //float percent_in_agreement = (float)largest_occurrance / (float)tot_occurrances;
   if ( !are_all_a_multiple_of_the_smallest || difference_list.size() < 3 || smallest_gcd < OBJECT_SIZE_MIN ) {
      smallest_gcd = INVALID_OBJECT_SIZE;
   }
#if ENABLE_POLICY_MAN_PRINTF
   if ( INVALID_OBJECT_SIZE != smallest_gcd ) {
      printf( "vm_policy_manager::determine_gcd_obj_size thinks object size = %zu Percent Agreement = %f Total Occ = %u are_all_a_multiple_of_the_smallest = %d\n",
            best_gcd, percent_in_agreement, tot_occurrances, are_all_a_multiple_of_the_smallest );
   }
#endif
   return smallest_gcd;
}

size_t vm_policy_manager::determine_region_access_vector_obj_size( const struct region_entry &region ) const {
   int max_score=0;
   size_t best_stride=1;
   for( unsigned stride=2; stride < m_config.region_size(); stride++ ) {
      int score=0;
      for( unsigned idx=0; idx < m_config.region_size(); idx+=stride ) {
          if( region.m_accessed.test(idx) ) {
              score++;
          } else {
              score--;
          }
      }
      if( score > max_score ) {
          max_score = score;
          best_stride = stride;
      }
   }
#if ENABLE_POLICY_MAN_PRINTF
   printf("#### core %u, 0x%8llx : best_stride = %u; score = %8d, bits set = %5zu => ",
         m_core_id, block_addr, best_stride, max_score, region.m_accessed.count() );
#endif
   return best_stride;
}

void vm_policy_manager::print( FILE* fp ) const {
    m_stats.print( fp );
    print_object_access_info( fp );
}

void vm_policy_manager::print_object_access_info( FILE* fp ) const {
   std::map< size_t, hit_vec_and_start_of_first_page >::const_iterator it = m_obj_size_to_hit_vector.begin();
   fprintf( fp,"vm_policy_manager\nnumber_of obj_sizes_found - %zu\n"
         , m_obj_size_to_hit_vector.size() );
   for ( ; it != m_obj_size_to_hit_vector.end(); ++it ) {
      fprintf(fp,"Object size %zu hotbytes: ", it->first );
      unsigned num_zero_count = 0;;
      for ( unsigned i = 0; i < it->second.vec.size(); ++i ) {
         if ( 0 == it->second.vec.at( i ) ) {
            ++num_zero_count;
            continue;
         } else if ( num_zero_count > 0 ) {
            fprintf(fp," %u zeros, ", num_zero_count );
            num_zero_count = 0;
         }
         fprintf(fp,"%u, ", it->second.vec.at( i ) );
      }
      if ( num_zero_count > 0 ) {
         fprintf(fp," %u zeros, ", num_zero_count );
      }
      fprintf(fp,"\n" );
   }
}

void vm_policy_manager::update_object_access_vector( const warp_inst_t& inst ) {
    if ( LOAD_OP == inst.op || STORE_OP == inst.op ) {
        size_t object_size = determine_gcd_obj_size( inst );
        const size_t page_size = m_config.get_line_sz();
        if ( object_size != INVALID_OBJECT_SIZE ) {
            for ( unsigned i = 0; i < inst.warp_size(); ++i ) {
                if ( inst.active( i ) ) {
                    if ( object_size != m_obj_size_to_hit_vector[ object_size ].vec.size() ) {
                        m_obj_size_to_hit_vector[ object_size ].vec.resize( object_size, 0 );
                        m_obj_size_to_hit_vector[ object_size ].start_of_first_page = ( inst.get_addr( i ) / page_size ) * page_size;
                    } else {
                        unsigned displacement = inst.get_addr( i ) - m_obj_size_to_hit_vector[ object_size ].start_of_first_page
                                - ( ( inst.get_addr( i ) - m_obj_size_to_hit_vector[ object_size ].start_of_first_page ) / object_size ) * object_size;
                        assert( displacement < object_size );
                        for ( unsigned i = 0; i < inst.data_size; ++i ) {
                            assert( displacement + i < object_size );
                            m_obj_size_to_hit_vector[ object_size ].vec[ displacement + i ]++;
                            //print_object_access_info( stdout );
                        }
                    }
                }
            }
        }
    }
}

vm_page_mapping_payload vm_policy_manager::get_no_translation( new_addr_type block_addr, size_t obj_size ) const {
#if ENABLE_POLICY_MAN_PRINTF
        printf(" NO_TRANSLATION (1)\n");
#endif
    virtual_page_creation_t new_page_definition;
    new_page_definition.m_type = VP_NON_TRANSLATED;
    new_page_definition.m_start = reinterpret_cast< void* >( block_addr );
    new_page_definition.m_end = reinterpret_cast< void* >( block_addr + m_config.get_line_sz() );
    return vm_page_mapping_payload( new_page_definition.m_type, new_page_definition );
}

vm_page_mapping_payload vm_policy_manager::select_tiled_mapping( new_addr_type block_addr, size_t obj_size ) {
    virtual_page_creation_t new_page_definition;
    if( obj_size <= m_config.m_word_size || INVALID_OBJECT_SIZE == obj_size ) {
        return get_no_translation( block_addr, obj_size );
    }

    if ( ( obj_size % m_l1_cache_line_size != 0 && obj_size > m_l1_cache_line_size )
            || ( m_l1_cache_line_size % obj_size != 0 && obj_size < m_l1_cache_line_size )
            || m_config.get_line_sz() % obj_size != 0 )
        return get_no_translation( block_addr, obj_size );

    unsigned objects_per_page = m_config.region_size() * m_config.m_word_size / obj_size;
    new_page_definition.m_start = reinterpret_cast< void* >( block_addr );
    new_page_definition.m_end = reinterpret_cast< void* >( block_addr + m_config.get_line_sz() );
    new_page_definition.m_field_size = m_config.m_word_size;
    new_page_definition.m_tile_width = objects_per_page;
    new_page_definition.m_type = VP_TILED;
    new_page_definition.m_object_size = obj_size;

    return vm_page_mapping_payload( new_page_definition.m_type, new_page_definition );
}

vm_page_mapping_payload vm_policy_manager::select_aligned_hot_mapping( new_addr_type block_addr, size_t object_size ) {
    virtual_page_creation_t new_page_definition;

    if( object_size <= m_config.m_word_size || INVALID_OBJECT_SIZE == object_size ) {
        return get_no_translation( block_addr, object_size );
    }

    // Currently the object size must be a multiple of the cache line size
    if ( ( object_size % m_l1_cache_line_size != 0 && object_size > m_l1_cache_line_size )
            || ( m_l1_cache_line_size % object_size != 0 && object_size < m_l1_cache_line_size )
            || m_config.get_line_sz() % object_size != 0 )
        object_size = ( object_size / m_l1_cache_line_size + 1 ) * m_l1_cache_line_size;
    unsigned objects_per_region = m_config.region_size() * m_config.m_word_size / object_size;
    unsigned accesses = find_hot_data( new_page_definition.m_hotbytes, object_size );
    new_page_definition.m_start = reinterpret_cast< void* >( block_addr );
    new_page_definition.m_end = reinterpret_cast< void* >( block_addr + m_config.get_line_sz() );
    if ( objects_per_region == 0 || accesses < m_config.m_threshold ) {
#if ENABLE_POLICY_MAN_PRINTF
        printf(" VP_NON_TRANSLATED (2)\n");
#endif
        new_page_definition.m_type = VP_NON_TRANSLATED;
    } else {
        new_page_definition.m_object_size = object_size;
        new_page_definition.m_page_size = m_config.get_line_sz();
        new_page_definition.m_type = VP_HOT_DATA_ALIGNED;
#if ENABLE_POLICY_MAN_PRINTF
        printf(" VP_HOT_DATA_ALIGNED : object_size = %u from %llx to %llx\n", object_size, block_addr, end_addr);
#endif
    }
    return vm_page_mapping_payload( new_page_definition.m_type, new_page_definition );
}

vm_page_mapping_payload vm_policy_manager::select_unaligned_hot_mapping( new_addr_type block_addr, size_t object_size ) {
    if( object_size <= m_config.m_word_size || INVALID_OBJECT_SIZE == object_size ) {
        return get_no_translation( block_addr, object_size );
    }

    assert( m_obj_size_to_hit_vector.find( object_size ) != m_obj_size_to_hit_vector.end() );
    virtual_page_creation_t new_page_definition;

    unsigned accesses = find_hot_data( new_page_definition.m_hotbytes, object_size );
    new_page_definition.m_start = reinterpret_cast< void* >( block_addr );
    new_page_definition.m_end = reinterpret_cast< void* >( block_addr + m_config.get_line_sz() );

    if( accesses < m_config.m_threshold ) {
#if ENABLE_POLICY_MAN_PRINTF
        printf(" VP_NON_TRANSLATED (2)\n");
#endif
        new_page_definition.m_type = VP_NON_TRANSLATED;
    } else {
#if ENABLE_POLICY_MAN_PRINTF
        printf(" VP_HOT_DATA_UNALIGNED : object_size = %u from %llx to %llx\n", object_size, block_addr, end_addr);
#endif
        new_page_definition.m_object_size = object_size;
        new_page_definition.m_cache_block_size = m_l1_cache_line_size;
        new_page_definition.m_start_of_first_page = m_obj_size_to_hit_vector[ object_size ].start_of_first_page;
        new_page_definition.m_page_size = m_config.get_line_sz();
        new_page_definition.m_type = VP_HOT_DATA_UNALIGNED;
    }
    return vm_page_mapping_payload( new_page_definition.m_type, new_page_definition );
}

unsigned vm_policy_manager::find_hot_data( boost::dynamic_bitset<>& hotmap, size_t object_size ) const {
    unsigned accesses=0;
    hotmap.resize( object_size, 0 );
    hotmap.reset();
    std::map< size_t, hit_vec_and_start_of_first_page >::const_iterator i = m_obj_size_to_hit_vector.find(object_size);
    if( i != m_obj_size_to_hit_vector.end() ) {
        const std::vector<unsigned> &profile = i->second.vec;
        std::vector<unsigned>::const_iterator b;
        unsigned n=0;
        unsigned consecutive_bytes = 0;
        for( b=profile.begin(); b!=profile.end(); b++, n++ ) {
            accesses +=*b;
            if( *b > 0 ) {
                consecutive_bytes++;
                hotmap[ n ] = 1;
            } else {
                // If the hot bytes are not word aligned....
                if ( consecutive_bytes > 0 ) {
                    for ( unsigned j = 0; j < m_config.m_word_size - consecutive_bytes % m_config.m_word_size; ++j ) {
                        hotmap[ n + j ] = 1;
                    }
                }
                consecutive_bytes = 0;
            }
        }
    }
    return accesses;
}

void vm_policy_manager::update_region_access_vector( warp_inst_t& inst, unsigned time ) {
    for ( unsigned i=0; i < inst.warp_size(); i++ ) {
        if ( inst.active( i ) ) {
            new_addr_type addr = inst.get_addr( i );
            unsigned idx=-1;
            enum cache_request_status status = m_tags.access( addr, time, idx, NULL );
            if( status == MISS ) {
#if ENABLE_POLICY_MAN_PRINTF
                printf("#### core %u, miss on address 0x%8llx\n", m_core_id, addr );
#endif
                m_tags.fill(idx,time);
            }
            region_entry &e = m_data[idx];
            unsigned offset = (addr / m_config.m_word_size) % m_config.region_size();
            e.access(offset);
        }
    }
}

void vm_policy_manager::handle_new_policy( vm_page_mapping_payload* best, new_addr_type block_addr, region_entry &e, shader_page_reordering_tlb& tlb ) {
    if( best ) {
        base_virtual_page* current_page = NULL;
        tlb.access( block_addr, current_page );
        if( current_page && *current_page == *best->get_virtualized_page() ) {
            m_stats.redundant_policy_event();
            delete best;
        } else if( m_config.m_update_policy == vm_manager_config::UPDATE_POLICY_USE_REGION_CACHE ) {
            // If we are using the region cache
            mem_fetch *mf = m_mf_allocator->alloc(block_addr,VM_POLICY_CHANGE_REQ,4,false,best);
            m_request_fifo.push_back(mf);
            e.m_pending_change=true;
            e.m_count = 0;
        } else if( m_config.m_update_policy == vm_manager_config::UPDATE_POLICY_DIRECT_TO_TLB ) {
            // Otherwise go straight to the TLB.
            m_tlb_update_queue.push_back( best );
        } else abort();
    }
}

// This might be a bit unrealistic, but for now lets assume that the policy manager knows the outcome of the
// TLB lookup for these instruction addresses and know what policy we currently think they have, that way we do not spam
// The interconnect with traffic based on the same decision happening over and over again.
void vm_policy_manager::access( warp_inst_t& inst, unsigned time, shader_page_reordering_tlb& tlb ) {
    if( !m_config.m_enabled || global_space != inst.space.get_type() ) return;

    // First update our access vectors
    update_region_access_vector( inst, time );
    update_object_access_vector( inst );

    vm_page_mapping_payload* best = NULL;
    size_t object_size = INVALID_OBJECT_SIZE;
    switch( m_config.m_object_size_policy ) {
    case vm_manager_config::OBJ_SIZE_POLICY_TYPE_GCD:
        // GCD finds the object size sheerly from the instruction
        object_size = determine_gcd_obj_size( inst );

        for ( unsigned i=0; i < inst.warp_size(); i++ ) {
            if ( inst.active( i ) ) {
                unsigned idx=-1;
                new_addr_type addr = inst.get_addr( i );
                new_addr_type block_addr = m_config.block_addr( addr );

                m_tags.probe( addr, idx );
                assert( idx >= 0 && idx < m_config.get_num_lines() );

                switch( g_translated_page_type ) {
                case VP_TILED:
                    best = new vm_page_mapping_payload( select_tiled_mapping( block_addr, object_size ) );
                    break;
                case VP_HOT_DATA_ALIGNED:
                    best =  new vm_page_mapping_payload( select_aligned_hot_mapping( block_addr, object_size ) );
                    break;
                case VP_HOT_DATA_UNALIGNED:
                    best =  new vm_page_mapping_payload( select_unaligned_hot_mapping( block_addr, object_size ) );
                    break;
                default:
                    printf( "Unknown Page Type\n" );
                    abort();
                }

                region_entry &e = m_data[idx];
                handle_new_policy( best, block_addr, e, tlb );
                m_stats.attemtped_policy_event( best, object_size );
            }
        }
        break;
    case vm_manager_config::OBJ_SIZE_POLICY_TYPE_BYTE_LINE_INTERPRETATION: {
        // Interpretation of the region access pattern looks at all the regions accessed by this instruction
        for ( unsigned i=0; i < inst.warp_size(); i++ ) {
            if ( inst.active( i ) ) {
                new_addr_type addr = inst.get_addr( i );
                new_addr_type block_addr = m_config.block_addr( addr );
                unsigned idx=-1;
                m_tags.probe( addr, idx );
                assert( idx >= 0 && idx < m_config.get_num_lines() );
                region_entry &e = m_data[idx];
                object_size = determine_region_access_vector_obj_size( e );

                switch( g_translated_page_type ) {
                case VP_TILED:
                    best =  new vm_page_mapping_payload( select_tiled_mapping( block_addr, object_size ) );
                    break;
                case VP_HOT_DATA_ALIGNED:
                    best =  new vm_page_mapping_payload( select_aligned_hot_mapping( block_addr, object_size ) );
                    break;
                case VP_HOT_DATA_UNALIGNED:
                    best =  new vm_page_mapping_payload( select_unaligned_hot_mapping( block_addr, object_size ) );
                    break;
                default:
                    printf( "Unknown Page Type\n" );
                    abort();
                }

                handle_new_policy( best, block_addr, e, tlb );
                m_stats.attemtped_policy_event( best, object_size );
            }
        }
    } break;
    default:
        printf( "vm_policy_manager - Unknown object detecton policy\n" );
        abort();
    }
}

vm_page_mapping_payload *vm_policy_manager::cycle()
{
    vm_page_mapping_payload *result = NULL;
    if( !m_config.m_enabled )
        return result;
    if( !m_tlb_update_queue.empty() ) {
        result = m_tlb_update_queue.front();
        m_tlb_update_queue.pop_front();
    }
    if( !m_request_fifo.empty() ) {
        mem_fetch *mf = m_request_fifo.front();
        if( !m_icnt->full( mf->get_data_size(), mf->get_is_write() ) ) {
            m_request_fifo.pop_front();
            m_icnt->push(mf);
        }
    }
    return result;
}

void vm_policy_manager::fill( mem_fetch *mf )
{
  assert( m_config.m_enabled );
  unsigned idx=-1;
  enum cache_request_status status = m_tags.probe(mf->get_addr(),idx);
  enum mem_access_type type = mf->get_access_type();
  vm_page_mapping_payload *mapping = dynamic_cast<vm_page_mapping_payload*>(mf->get_payload());
  if( status == HIT ) {
      region_entry &e = m_data[idx];
      if( type == VM_POLICY_CHANGE_ACK_RC ) {
          e.m_pending_change=false; // change no longer pending
      } else if( type == VM_POLICY_CHANGE_REQ_RC ) {
          // TODO: we should change our policy to match region cache
      } else
          abort();
  } else {
      if( type == VM_POLICY_CHANGE_ACK_RC ) {
          // TODO: measure how often this happens (should have stalled on request)
      } else if( type == VM_POLICY_CHANGE_REQ_RC ) {
          // TODO: we should change our policy to match region cache
      } else
          abort();
  }
  m_tlb_update_queue.push_back( mapping );
}

enum cache_request_status virtual_policy_cache::access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) {
    bool wr = mf->get_is_write();
    enum mem_access_type type = mf->get_access_type();
    bool evict = (type == GLOBAL_ACC_W); // evict a line that hits on global memory write

    new_addr_type block_addr = m_config.block_addr(addr);
    m_stats.access_event( block_addr );

    unsigned cache_index = (unsigned)-1;

//     enum cache_request_status status = m_tag_array->probe(block_addr,cache_index);
    enum cache_request_status status = m_tag_array->probe(addr,cache_index);   // TODO: Tayler - Needed actual address for word access... not just block address

    enum cache_request_status status_overwrite = HIT;
    // You miss in the $ if we are using remapping and the policy on the line does not match your current policy
    vm_page_mapping_payload* page = dynamic_cast< vm_page_mapping_payload* >( mf->get_payload() );
    if ( ( status == HIT || status == HIT_RESERVED ) && DYNAMIC_TRANSLATION == g_translation_config &&
       ( ( page && page->get_virtualized_page() && m_tag_array->get_block( cache_index ).m_policy != *page ) ) )
    {
       //printf("Remap miss caused by address %p\ncache_index %u\nExisting:\n", (void*)addr, cache_index);
       //m_tag_array->get_block( cache_index ).m_policy.get_region_tile_definition().print( stdout );
       //printf( "New:\n" );
       //page->print(stdout);
       status = MISS;
       status_overwrite = MISS;
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
        } else {
            cache_block_t evicted( m_config.get_line_sz() );
            bool wb= false;
            //status = m_tag_array->access(block_addr,time,cache_index, wb, evicted); // update LRU state
            status = m_tag_array->access(addr,time,cache_index, wb, evicted, mf); // TODO: Tayler - Needed actual address for word access... not just block address

            // Kind of a hack place to put this but we are less than 2 days from the deadline and just trying to make this thing work
            if ( MISS == status && DYNAMIC_TRANSLATION == g_translation_config && m_config.m_alloc_policy == ON_MISS ) {
               if ( mf->get_payload() ) {
                  m_tag_array->get_block( cache_index ).m_policy = *(dynamic_cast< vm_page_mapping_payload* >( mf->get_payload() ));
               }
            }
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
                status = m_tag_array->access(addr,time,cache_index,wb,evicted, mf);
                m_stats.miss_event( addr, evicted, cache_index, mf, status );
                m_mshrs.add(block_addr,mf);
                do_miss = true;
            } else if ( !mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size) ) {
                status = m_tag_array->access(addr,time,cache_index,wb,evicted, mf);
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
               // Kind of a hack place to put this but we are less than 2 days from the deadline and just trying to make this thing work
               if ( MISS == status_overwrite && DYNAMIC_TRANSLATION == g_translation_config && m_config.m_alloc_policy == ON_MISS ) {
                  if ( mf->get_payload() ) {
                     //printf("Payload write proceeding address %p\ncache_index %u\nExisting:\n", (void*)addr, cache_index);
                     //m_tag_array->get_block( cache_index ).m_policy.get_region_tile_definition().print( stdout );
                     //printf( "New:\n" );
                     //page->print(stdout);
                     m_tag_array->get_block( cache_index ).m_policy = *(dynamic_cast< vm_page_mapping_payload* >( mf->get_payload() ));
                  }
               }
               return MISS;
            }

        }
    }

    return RESERVATION_FAIL;
}

enum cache_request_status evict_on_write_cache::access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) {
     assert( mf->get_data_size() <= m_config.get_line_sz());

     bool wr = mf->get_is_write();
     bool isatomic = mf->isatomic();
     enum mem_access_type type = mf->get_access_type();
     bool evict = (type == GLOBAL_ACC_W); // evict a line that hits on global memory write

     new_addr_type block_addr = m_config.block_addr(addr);
     m_stats.access_event( block_addr );

     unsigned cache_index = (unsigned)-1;
     
//	std::bitset<NUM_SECTORED> sector_mask = mf->get_sector_mask();

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
             //assert( block.m_status != MODIFIED ); // fails if block was allocated by a ld.local and now accessed by st.global

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
             if (m_tag_array->get_stats().m_access_stream.size() > 0) {
                m_tag_array->get_stats().m_access_stream.back().set_access_type( type );
                m_tag_array->get_stats().m_access_stream.back().set_request_status( status );
             }

             m_stats.hit_event( addr, evicted, cache_index, mf, status );

             if ( wr ) {
                 assert( type == LOCAL_ACC_W || /*l2 only*/ type == L1_WRBK_ACC );
                 // treated as write back...
                 cache_block_t &block = m_tag_array->get_block(cache_index);
                 block.m_status = MODIFIED;
	     } else if ( isatomic ) {
                 assert( type == GLOBAL_ACC_R ); 
                 // treated as write back...
                 cache_block_t &block = m_tag_array->get_block(cache_index);
                 block.m_status = MODIFIED;  // mark line as dirty 
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
             if ( m_config.enable_cache_access_dump ) {
                 m_tag_array->get_stats().m_access_stream.push_back(
                         access_stream_entry( addr, type, status, mf ) );
             }
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

                 if (m_tag_array->get_stats().m_access_stream.size() > 0) {
                    m_tag_array->get_stats().m_access_stream.back().set_access_type( type );
                    m_tag_array->get_stats().m_access_stream.back().set_request_status( status );
                 }
                 m_stats.miss_event( addr, evicted, cache_index, mf, status );
                 m_mshrs.add(block_addr,mf);
                 do_miss = true;
             } else if ( !mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size) ) {
                 status = m_tag_array->access( addr,time, cache_index, wb, evicted, mf );
                 if (m_tag_array->get_stats().m_access_stream.size() > 0) {
                    m_tag_array->get_stats().m_access_stream.back().set_access_type( type );
                    m_tag_array->get_stats().m_access_stream.back().set_request_status( status );
                 }
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

void high_locality_protected_cache::print( FILE *fp, unsigned &accesses, unsigned &misses ) {
    m_tag_array->print_protected_info( fp );
    evict_on_write_cache::print( fp, accesses, misses );

    fprintf( stdout, "weighted_potential_protected_lines=%zu\n", m_dynamic_detection_pcs_to_protect.size() );
    for ( std::map< new_addr_type, unsigned >::const_iterator it = m_dynamic_detection_pcs_to_protect.begin(); it != m_dynamic_detection_pcs_to_protect.end(); ++it ) {
        const char* file = NULL;
        unsigned line = 0;
        get_ptx_source_info( it->first, file, line );
        fprintf( stdout, "%u w=%u, ", line, it->second );
    }
    fprintf(stdout, "\n");

    fprintf( stdout, "dynamically_protected_source_lines=%zu\n", m_reduced_list_to_protect.size() );
    for ( std::list< new_addr_type >::const_iterator it = m_reduced_list_to_protect.begin(); it != m_reduced_list_to_protect.end(); ++it ) {
        const char* file = NULL;
        unsigned line = 0;
        get_ptx_source_info( *it, file, line );
        fprintf( stdout, "%u, ", line );
    }
    fprintf(stdout, "\n");
}

enum cache_request_status high_locality_protected_cache::access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events, warp_inst_t &inst ) {
    const char* file = NULL;
    unsigned line = 0;
    get_ptx_source_info( inst.pc, file, line );

    // Locality detection
    unsigned cache_index = 0xDEADBEEF;
    priority_cache_block block_to_be_evicted( m_config.get_line_sz() );
    if ( m_high_loc_config.get_type() == DYNAMIC_HIGH_LOCALITY_CACHE ) {
        cache_request_status status = m_tag_array->probe(addr,cache_index);
        if ( MISS == status ) {
            assert( cache_index != 0xDEADBEEF );
            block_to_be_evicted = dynamic_cast< priority_cache_block& >( m_tag_array->get_block( cache_index ) );
        }
    }

    if ( ( m_high_loc_config.get_type() == STATIC_HIGH_LOCALITY_CACHE && m_high_loc_config.does_line_have_high_locality( file, line ) )
        || ( m_high_loc_config.get_type() == DYNAMIC_HIGH_LOCALITY_CACHE && does_have_dynamic_high_locality( inst.pc ) ) ) {
        unsigned cache_index = 0xDEADBEEF;
        unsigned num_pro = 0;
        cache_request_status status = m_tag_array->probe_special(addr,cache_index, num_pro);
        //cache_request_status status = m_tag_array->probe(addr,cache_index);
        //if ( MISS == status ) {
        if ( RESERVATION_FAIL != status ) {
            assert( cache_index != 0xDEADBEEF );

            // warp protection list
            if ( hack_use_warp_prot_list ) {
                priority_cache_block& block = dynamic_cast< priority_cache_block& >( m_tag_array->get_block( cache_index ) );
                if ( block.m_warps_protecting_line.find( inst.warp_id() ) == block.m_warps_protecting_line.end() )
                    block.m_warps_protecting_line[ inst.warp_id() ] = 1;
            } else {
                // Expiring priorities
                m_tag_array->protect_line( cache_index, inst.warp_id() );

            }
        }
    }
    cache_request_status access_status = evict_on_write_cache::access( addr, mf, time, events );
    cache_request_status probe_status = m_tag_array->probe(addr,cache_index);

    if ( m_high_loc_config.get_type() == DYNAMIC_HIGH_LOCALITY_CACHE) {
        bool own_hit = false;

        if (MISS == access_status) {
            assert( cache_index != 0xDEADBEEF );
            priority_cache_block& new_block = dynamic_cast< priority_cache_block& >( m_tag_array->get_block( cache_index ) );
            new_block.m_source_pc = inst.pc;
            new_block.m_source_warp = inst.warp_id();

            if ( block_to_be_evicted.m_source_pc != 0 ) {
                locality_detection_handle_eviction( block_to_be_evicted.m_block_addr, block_to_be_evicted.m_source_pc, block_to_be_evicted.m_source_warp );
            }
            if ( handle_victim_cache_probe( addr, inst ) ) {
                events.push_back(VC_HIT);
            }
            set_limited_protected_lines();
        } else if (HIT == access_status 
                   && mf->get_access_type() != GLOBAL_ACC_W) {
            assert( HIT == probe_status );
            assert( cache_index != 0xDEADBEEF );
            own_hit = dynamic_cast< priority_cache_block& >(
                        m_tag_array->get_block(cache_index) )
                      .m_source_warp == inst.warp_id();
        }
        // Evict writes from the VC as well
        if (mf->get_access_type() == GLOBAL_ACC_W) {
            if ( m_victim_tag_arrays.size() > 0 ) {
                for (unsigned i = 0; i < m_victim_tag_arrays.size(); ++i) {
                    unsigned cache_index = 0xDEADBEEF;
                    cache_request_status status
                        = m_victim_tag_arrays[i]->probe( addr, cache_index );
                    assert( RESERVATION_FAIL != status );
                    if ( HIT == status ) {
                        assert(0xDEADBEEF != cache_index);
                        m_victim_tag_arrays[i]->get_block(cache_index).m_status = INVALID;
                    }
                }
            } else if ( m_victim_tag_array ) {
                unsigned cache_index = 0xDEADBEEF;
                cache_request_status status
                    = m_victim_tag_array->probe( addr, cache_index );
                assert( RESERVATION_FAIL != status );
                if ( HIT == status ) {
                    assert(0xDEADBEEF != cache_index);
                    m_victim_tag_array->get_block(cache_index).m_status = INVALID;
                }

            }
        }
        m_point_system->cache_access( inst.warp_id(), access_status, own_hit );
    }
    return access_status;
}

bool high_locality_protected_cache::handle_victim_cache_probe( new_addr_type addr, const warp_inst_t &inst ) {
    bool hits = false;
    if (m_victim_tag_arrays.size() > 0) {
        unsigned cache_index = 0xDEADBEEF;
        const cache_request_status status = m_victim_tag_arrays[inst.warp_id()]->probe( addr, cache_index );
        if ( HIT == status ) {
            hits = true;
            new_addr_type src_pc
                = dynamic_cast< victim_cache_block& >( m_victim_tag_arrays[inst.warp_id()]->get_block( cache_index ) ).m_source_pc;
            assert( src_pc != 0 );
            if ( ptx_fetch_inst( src_pc )->is_load() ) {
                ++m_dynamic_detection_pcs_to_protect_per_warp[inst.warp_id()][ src_pc ];
            }
        }
    } else if ( m_victim_tag_array ) {
        unsigned cache_index = 0xDEADBEEF;
        const cache_request_status status = m_victim_tag_array->probe( addr, cache_index );
        if ( HIT == status ) {
            hits = true;
            new_addr_type src_pc
                = dynamic_cast< victim_cache_block& >( m_victim_tag_array->get_block( cache_index ) ).m_source_pc;
            assert( src_pc != 0 );
            if ( ptx_fetch_inst( src_pc )->is_load() ) {
                ++m_dynamic_detection_pcs_to_protect[ src_pc ];
            }
        }
    } else {
        if ( m_victim_cache_hash.find( addr ) != m_victim_cache_hash.end() ) {
            hits = true;
            if ( ptx_fetch_inst( m_victim_cache_hash[ addr ].source_pc )->is_load() ) {
                m_dynamic_detection_pcs_to_protect[ m_victim_cache_hash[ addr ].source_pc ]++;
            }
        }
    }

    const bool load_hits = hits && inst.is_load();
    if ( load_hits ) {
        if (m_dynamic_detection_pcs_to_protect_per_warp.size() > 0) {
            m_dynamic_detection_pcs_to_protect_per_warp[inst.warp_id()][inst.pc]++;
        } else {
            m_dynamic_detection_pcs_to_protect[ inst.pc ]++;
        }
        m_point_system->vc_hit(inst.warp_id());
    }
    return load_hits;
}

void high_locality_protected_cache::locality_detection_handle_eviction( new_addr_type block_addr, new_addr_type pc, unsigned warp_id ) {
    // if the new block is not already in our system, then handle potential deletion and add the block
    if ( m_high_loc_config.get_victim_cache_scope() == ALL_WARPS_VICTIM_CACHE_SCOPE || ( m_high_loc_config.get_victim_cache_scope() == ONE_WARP_VICTIM_CACHE_SCOPE && 0 == warp_id ) ) {
        if ( m_victim_tag_arrays.size() > 0 ) {
            unsigned cache_index = 0xDEADBEEF;
            cache_request_status status
                = m_victim_tag_arrays[warp_id]->access( block_addr, gpu_sim_cycle + gpu_tot_sim_cycle, cache_index, NULL );
            assert( RESERVATION_FAIL != status );
            if ( MISS == status ) {
                m_victim_tag_arrays[warp_id]->fill( block_addr, gpu_sim_cycle + gpu_tot_sim_cycle );
                status = m_victim_tag_arrays[warp_id]->probe( block_addr, cache_index );
                assert( status == HIT && 0xDEADBEEF != cache_index );
                dynamic_cast< victim_cache_block& >( m_victim_tag_arrays[warp_id]->get_block( cache_index ) ).m_source_pc = pc;
            } else {
                // Update the LRU stack
                assert( HIT == status );
                m_victim_tag_arrays[warp_id]->access( block_addr, gpu_sim_cycle + gpu_tot_sim_cycle, cache_index, NULL );
            }

        } else if ( m_victim_tag_array ) {
            unsigned cache_index = 0xDEADBEEF;
            cache_request_status status
                = m_victim_tag_array->access( block_addr, gpu_sim_cycle + gpu_tot_sim_cycle, cache_index, NULL );
            assert( RESERVATION_FAIL != status );
            if ( MISS == status ) {
                m_victim_tag_array->fill( block_addr, gpu_sim_cycle + gpu_tot_sim_cycle );
                status = m_victim_tag_array->probe( block_addr, cache_index );
                assert( status == HIT && 0xDEADBEEF != cache_index );
                dynamic_cast< victim_cache_block& >( m_victim_tag_array->get_block( cache_index ) ).m_source_pc = pc;
            }
        } else {
            if ( m_victim_cache_hash.find( block_addr ) == m_victim_cache_hash.end() ) {
                if ( m_victim_cache_deletion_order.size() >= m_high_loc_config.get_num_victim_entries() ) {
                    m_victim_cache_hash.erase( m_victim_cache_deletion_order.front() );
                    m_victim_cache_deletion_order.pop_front();
                }
                m_victim_cache_hash[ block_addr ].source_pc = pc;
                m_victim_cache_hash[ block_addr ].cycles_until_detection = gpu_sim_cycle + gpu_tot_sim_cycle;
                m_victim_cache_deletion_order.push_back( block_addr );
            }
            assert( m_victim_cache_deletion_order.size() == m_victim_cache_hash.size() && m_victim_cache_hash.size() <= m_high_loc_config.get_num_victim_entries() );
        }

    }
}

typedef std::pair< new_addr_type, unsigned > mypair;

bool cmp_seconds(const mypair &lhs, const mypair &rhs) {
    return lhs.second > rhs.second;
}

void high_locality_protected_cache::print_victim_cache_results() const
{
    printf("High locality cache VC Stats\n");
    if (VC_PER_WARP) {
        int count = 0;
        for (std::vector< std::map< new_addr_type, unsigned > >::const_iterator iter = m_dynamic_detection_pcs_to_protect_per_warp.begin(); iter != m_dynamic_detection_pcs_to_protect_per_warp.end(); ++iter ) {
            printf("Warp %d\n", count);
            for (std::map< new_addr_type, unsigned >::const_iterator iter2 =  iter->begin();
                    iter2 != iter->end(); ++iter2) {
                ptx_print_insn( iter2->first, stdout);
                printf("count=%d\n", iter2->second);
            }
            ++count;
        }

    } else {
        for (std::map< new_addr_type, unsigned >::const_iterator iter =  m_dynamic_detection_pcs_to_protect.begin();
                iter !=m_dynamic_detection_pcs_to_protect.end(); ++iter) {
            ptx_print_insn( iter->first, stdout);
            printf("count=%d\n", iter->second);
        }
    }
}

void high_locality_protected_cache::cycle()
{
    if(m_core_id == 0) {
        //DO_EVERY(200, m_point_system->print());
    }
    evict_on_write_cache::cycle();
    m_tag_array->cycle();
}

bool high_locality_protected_cache::should_whitelist_warp( unsigned warp_id ) const {
    return m_tag_array->should_whitelist_warp( warp_id );
}

void high_locality_protected_cache::set_limited_protected_lines() {
    std::vector< mypair > vector( m_dynamic_detection_pcs_to_protect.begin(), m_dynamic_detection_pcs_to_protect.end() );
    std::sort( vector.begin(), vector.end(), cmp_seconds );
    m_reduced_list_to_protect.clear();
    for ( std::vector< mypair >::const_iterator it = vector.begin(); it != vector.end(); ++it ) {
        if ( it->second > m_high_loc_config.get_weight_to_protect() ) {
            m_reduced_list_to_protect.push_back( it->first );
        }
    }
}

bool high_locality_protected_cache::does_have_dynamic_high_locality( new_addr_type pc ) const {
    for ( std::list< new_addr_type >::const_iterator it = m_reduced_list_to_protect.begin(); it != m_reduced_list_to_protect.end(); ++it ) {
        if ( *it == pc ) {
            return true;
        }
    }
    return false;
}

void priority_tag_array::cycle() {
    for ( std::vector< priority_cache_block * >::const_iterator it = m_lines.begin(); it != m_lines.end(); ++it ) {
        if ( (*it)->m_cycles_protected > 0 ) {
            assert( get_protection_type() == CYCLES_PROTECTION_TYPE );
            --(*it)->m_cycles_protected;
            if ( (*it)->m_cycles_protected == 0 ) {
                --m_num_protected_lines;
                if ( !hack_use_warp_prot_list ) {
                    assert( (*it)->m_warps_protecting_line.size() == 1 );
                    --m_lines_protected_per_warp[ ( ( *it )->m_warps_protecting_line.begin() )->first ].second;
                    (*it)->m_warps_protecting_line.clear();
                }
            }
        }
    }
    assert( m_num_protected_lines <= m_lines.size() );
}

void high_locality_protected_cache::flush()
{
    if ( m_victim_tag_arrays.size() > 0 ) {
        std::vector<victim_tag_array*>::iterator iter
            = m_victim_tag_arrays.begin();
        while ( iter != m_victim_tag_arrays.end() ) {
            (*iter)->flush();
            ++iter;
        }
    } else {
        m_victim_tag_array->flush();
    }
    evict_on_write_cache::flush();
}

bool priority_tag_array::should_whitelist_warp( unsigned warp_id ) const {
    const unsigned num_warps_to_proceed = m_stall_override_factor == 0 ? m_lines_protected_per_warp.size()
            : m_num_protected_lines / m_stall_override_factor;
    if ( m_num_protected_lines == 0 ) {
        return true;
    }
    std::vector< count_warp_id_pair > sorted_vector( m_lines_protected_per_warp.begin(), m_lines_protected_per_warp.end() );
    std::sort( sorted_vector.begin(), sorted_vector.end(), cmp_seconds );
    for ( unsigned i = 0; i < num_warps_to_proceed; ++i ) {
       if ( warp_id == sorted_vector[ i ].first ) {
           return true;
       }
    }
    return false;
}

//enum cache_request_status priority_tag_array::probe( new_addr_type addr, unsigned &idx ) const {
enum cache_request_status priority_tag_array::probe_special( new_addr_type addr, unsigned &idx, unsigned& num_protected ) const {
    assert( m_config.m_write_policy == READ_ONLY );
    unsigned set_index = m_config.set_index(addr);
    new_addr_type tag = m_config.tag(addr);

    unsigned invalid_line = (unsigned)-1;
    unsigned valid_line = (unsigned)-1;
    unsigned valid_timestamp = (unsigned)-1;

    unsigned num_reserved = 0;
    num_protected = 0;
    bool all_unusable = true;

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

        bool is_protected;
        if ( get_protection_type() == CYCLES_PROTECTION_TYPE ) {
            is_protected = ( !hack_use_warp_prot_list &&  dynamic_cast< priority_cache_block* >( line )->m_cycles_protected > 0 )
                            || ( hack_use_warp_prot_list && dynamic_cast< priority_cache_block* >( line )->m_warps_protecting_line.size() > 0 );
        } else if ( get_protection_type() == ACCESSES_PROTECTION_TYPE ) {
            is_protected = dynamic_cast< priority_cache_block* >( line )->m_accesses_protected > 0;
        } else {
            abort();
        }


        if ( is_protected )++num_protected;
        if ( line->m_status == RESERVED )++num_reserved;

        if ( line->m_status != RESERVED && !is_protected ) {
            all_unusable = false;
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
                }
            }
        }
    }

    if ( all_unusable ) {
        assert( m_config.m_alloc_policy == ON_MISS );
        if ( num_protected == m_config.m_assoc ) {
            //printf( "RESERVATION_FAIL all protected num_reserved=%u\n", num_reserved );
        }
        //printf( "RESERVATION_FAIL num_protected=%u, num_reserved=%u\n", num_reserved, num_protected );
        return RESERVATION_FAIL; // miss and not enough space in cache to allocate on miss
    }

    if ( invalid_line != (unsigned)-1 ) {
        idx = invalid_line;
    } else if ( valid_line != (unsigned)-1) {
        idx = valid_line;
    } else abort(); // if an unreserved block exists, it is either invalid or replaceable

    return MISS;
}

void high_locality_cache_config::init( const char* config_str /*= NULL*/ ) {
    if ( !config_str ) config_str = m_config_string;
    assert( config_str );
    // Right now limiting this to 2 files with 5 lines maximum in each
    const unsigned STRLEN = 255;
    const unsigned MAX_LINES = 6;
    char file_1[ STRLEN ];
    unsigned lines_file_1[ MAX_LINES ];
    memset( lines_file_1, 0x0, MAX_LINES * sizeof( unsigned ) );
    char file_2[ STRLEN ];
    unsigned lines_file_2[ MAX_LINES ];
    memset( lines_file_2, 0x0, MAX_LINES * sizeof( unsigned ) );

    if ( strncmp( config_str, "dynamic", 7 ) == 0 ) {
        m_type = DYNAMIC_HIGH_LOCALITY_CACHE;
        sscanf( config_str + 7, ",%u,%u,%u,%u", &m_num_victim_entries, &m_victim_cache_scope, &m_weight_to_protect, &m_so_factor );
        return;
    }

    const char* first_parm = strchr( m_config_string, ',');
    if ( first_parm == NULL ) {
        assert( strcmp( config_str, "none" ) == 0 );
        return;
    }

    size_t size_of_translation_str =  (size_t)first_parm - (size_t)m_config_string;
    assert( size_of_translation_str < sizeof( file_1 ) );
    memcpy( file_1, m_config_string, size_of_translation_str );
    file_1[ size_of_translation_str ] = '\0';

    char config_2[ STRLEN ];
    config_2[ 0 ] = '\0';
    sscanf( first_parm+1,"%u,%u,%u,%u,%u,%u:%s",
                    &lines_file_1[ 0 ], &lines_file_1[ 1 ], &lines_file_1[ 2 ], &lines_file_1[ 3 ],
                    &lines_file_1[ 4 ], &lines_file_1[ 5 ], config_2 );

    if ( strlen( file_1 ) > 0 ) {
        for ( unsigned i = 0; i < MAX_LINES; ++i ) {
            if ( lines_file_1[ i ] != 0 ) {
                m_highlocality_source_lines[ file_1 ].insert( lines_file_1[ i ] );
            }
        }
    }

    if ( strlen( config_2 ) > 0 ) {
        const char* first_parm = strchr( config_2, ',');
        size_t size_of_translation_str =  (size_t)first_parm - (size_t)config_2;
        assert( size_of_translation_str < sizeof( file_2 ) );
        memcpy( file_2, config_2, size_of_translation_str );
        file_2[ size_of_translation_str ] = '\0';

        sscanf( first_parm+1,"%u,%u,%u,%u,%u,%u",
                &lines_file_1[ 0 ], &lines_file_1[ 1 ], &lines_file_1[ 2 ], &lines_file_1[ 3 ],
                &lines_file_1[ 4 ], &lines_file_1[ 5 ] );


        if ( strlen( file_2 ) > 0 ) {
            for ( unsigned i = 0; i < MAX_LINES; ++i ) {
                if ( lines_file_2[ i ] != 0 ) {
                    m_highlocality_source_lines[ file_2 ].insert( lines_file_2[ i ] );
                }
            }
        }
    }
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

