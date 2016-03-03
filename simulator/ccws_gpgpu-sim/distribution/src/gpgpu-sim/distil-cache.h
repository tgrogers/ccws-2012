#ifndef _DISTILL_CACHE_H_
#define _DISTILL_CACHE_H_

#include "gpu-cache.h"



enum distill_cache_request_status {
    LOC_HIT,	// Cache block is in LOC
    LOC_HIT_RESERVED,	// Cache block has been reserved in LOC
    WOC_HIT,	// Word is in the WOC
    HOLE_MISS,	// Word is not in the WOC, but other words from the requested line are 
    LINE_MISS,	// Miss in the LOC and WOC, line requested from DRAM
    DISTILL_RESERVATION_FAIL
};

enum sector_cache_request_status {
    SECTOR_HIT,
    SECTOR_HIT_RESERVED,// Word has already been request and reserved
    SECTOR_MISS,	 	// Line is not in the cache
	SECTOR_WORD_MISS,	// Line is in the cache, but the word is not
    SECTOR_RESERVATION_FAIL,
    SECTOR_FAIL_PROTECTION
};

typedef struct woc_info {
	woc_info() : tag(0), word_id(0), head_bit(false), state(INVALID) {}
	new_addr_type tag;
	unsigned word_id;
	bool head_bit;
	enum cache_block_state state;	
}woc_info;

struct woc_cache_block_t : public cache_block_t {
   woc_cache_block_t( unsigned line_size )
       : cache_block_t( line_size ) {
      m_footprint.reset();
      m_valid_words.reset();
   }

	virtual ~woc_cache_block_t() {}
	void allocate( new_addr_type tag, new_addr_type block_addr, unsigned word_id, unsigned time ) {
		cache_block_t::allocate( tag, block_addr, time );
		m_valid_words.set(word_id); // At least this word will be returned on fill
    }

    virtual void fill( unsigned time )
    {
        assert( m_status == RESERVED );
        m_status=VALID;
        m_fill_time=time;
    }
   void word_used(unsigned word_num){
      m_footprint.set(word_num);
   }

   bool is_word_used(unsigned word_num){
      if(m_footprint[word_num]) return true;
      return false;
   }

   void set_valid_words(std::bitset<8> t) { m_valid_words = t; }

   // All temporarily static - Probably should be a derived class for L2 Distillation
   std::bitset<8> m_footprint;     // Which words per block are used
   std::bitset<8> m_valid_words;   // Used by L1D cache if Distillation cache is active
   // Last lines in each set correspond to a WOC (not LOC)
   //std::map<new_addr_type, woc_info> WOC; // Word organized cache -> first = Tag, second = word_info
   woc_info WOC[8]; // For L2 -- Holds
};


// Sectored cache_block
//#define NUM_SECTORED 8
struct sectored_cache_block_t : public cache_block_t {
	// Valid = Some word in this block is valid
	// Reserved = No words are valid yet, but one or more are reserved. 
	sectored_cache_block_t(unsigned line_size)
		: cache_block_t(line_size) {
		//m_valid_words.reset();
		for(unsigned i=0; i<NUM_SECTORED; i++){
			word_state[i] = INVALID;
		}
	}

	void allocate( new_addr_type tag, new_addr_type block_addr, unsigned word_id, unsigned time ){
        m_tag=tag;
        m_block_addr=block_addr;
        m_alloc_time=time;
        m_last_access_time=time;
        m_fill_time=0;

		// Only update state if in INVALID state
		if(m_status == INVALID)
			m_status = RESERVED;
		if(word_state[word_id] == INVALID)
			word_state[word_id] = RESERVED;
    }

    void fill(unsigned word_id, unsigned time ){
        assert( m_status == RESERVED || m_status == VALID );
        m_status=VALID;
        m_fill_time=time;
		word_state[word_id] = VALID;
    }

	//std::bitset<8> m_valid_words;
	cache_block_state word_state[NUM_SECTORED];
};

class mem_div_cache : public base_tag_array {
public:
	mem_div_cache( const cache_config &config ) : base_tag_array( config,0,0,false )
    {
	    for ( unsigned i = 0; i < config.get_num_lines(); ++i ) {
            m_lines[ i ] = new woc_cache_block_t( config.get_line_sz() );
        }
        m_aux_stats = new struct stats[ config.get_num_lines() ];
    }
	virtual ~mem_div_cache()
    {
	    for ( unsigned i = 0; i < m_config.get_num_lines(); ++i ) {
            delete m_lines[ i ];
        }
        delete [] m_aux_stats;
    }

	enum cache_request_status access( new_addr_type addr, unsigned time, bool is_divergent, bool is_load_useful);

	bool is_this_load_useful( new_addr_type addr, shader_core_stats& shader_stats );

private:
    struct stats {
        stats() {
            div_streak_length=0;
            undiv_streak_length=0;
        }
        unsigned div_streak_length;
        unsigned undiv_streak_length;
    };
    struct stats *m_aux_stats;

    unsigned num_div_streak;
    unsigned sum_div_streak;
    unsigned num_undiv_streak;
    unsigned sum_undiv_streak;
};


class sector_tag_array : public base_tag_array {
public:
    sector_tag_array( const cache_config &config, int core_id, int type_id, bool enable_access_stream_dump ) : base_tag_array( config,core_id,type_id, enable_access_stream_dump ) {
        m_lines.resize( config.get_num_lines(), NULL );
        for ( unsigned i = 0; i < config.get_num_lines(); ++i ) {
            base_tag_array::m_lines[ i ] = m_lines[ i ] = new sectored_cache_block_t( config.get_line_sz() );
        }
    }

    virtual ~sector_tag_array() {
        for ( unsigned i = 0; i < m_config.get_num_lines(); ++i ) {
            delete m_lines[ i ];
        }
    }

    //cache_block_state get_block_status(sectored_cache_block_t *block, unsigned word_id) const {
	cache_block_state get_block_status(unsigned index, unsigned word_id) const {
		return m_lines[index]->word_state[word_id];
    }
	
	sectored_cache_block_t &get_block(unsigned index){
		return *m_lines[index];
	}

	enum sector_cache_request_status probe( new_addr_type addr, unsigned &idx, bool &all_reserved, std::bitset<NUM_SECTORED> mask) const;
	enum sector_cache_request_status access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted, std::bitset<NUM_SECTORED> mask);
	//enum sector_cache_request_status access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted);

	//void fill( new_addr_type addr, unsigned time );
	//void fill( unsigned idx, unsigned time );

	//sectored_cache_block_t &get_block(unsigned idx) const { return *m_lines[ idx ]; }

    std::vector< sectored_cache_block_t * > m_lines;
};

class distill_tag_array : public base_tag_array {
public:
	distill_tag_array( const cache_config &config, int core_id, int type_id, bool enable_access_stream_dump ) : base_tag_array( config,core_id,type_id,enable_access_stream_dump ) {
	    m_lines.resize( config.get_num_lines(), NULL );
	    for ( unsigned i = 0; i < config.get_num_lines(); ++i ) {
	        base_tag_array::m_lines[ i ] = m_lines[ i ] = new woc_cache_block_t( config.get_line_sz() );
        }
	}

	virtual ~distill_tag_array() {
	    for ( unsigned i = 0; i < m_config.get_num_lines(); ++i ) {
            delete m_lines[ i ];
        }
	}
	// Need new probe/access functions for the distill cache
	enum distill_cache_request_status probe( new_addr_type addr, unsigned &idx, unsigned &woc_idx ) const;
	enum distill_cache_request_status access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted, cache_block_t &evicted_woc_block);
	virtual enum cache_request_status access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted);

	void fill( new_addr_type addr, unsigned time );
	void fill( unsigned idx, unsigned time );

	woc_cache_block_t &get_block(unsigned idx) const { return *m_lines[ idx ]; }

	void evict_words(new_addr_type addr, unsigned idx); // Invalidate words from WOC

private:
	void save_used_words(new_addr_type addr, unsigned idx);

	std::vector< woc_cache_block_t * > m_lines;
};

/*
*	Line Distillation Cache
*	Derived from the following paper
*	http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4147665
*/

class distill_cache : public evict_on_write_cache {
public:
	distill_cache( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport, 
                mem_fetch_allocator *mfcreator, enum mem_fetch_status status, bool enable_access_dump )
    : evict_on_write_cache(name,config,core_id,type_id,memport,mfcreator,status,enable_access_dump) {
	    evict_on_write_cache::m_tag_array = m_tag_array = new distill_tag_array( config,core_id,type_id, enable_access_dump );
	}

	virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ){

		bool wr = mf->get_is_write();
//		enum mem_access_type type = mf->get_access_type();
//		bool evict = (type == GLOBAL_ACC_W); // evict a line that hits on global memory write

		new_addr_type block_addr = m_config.block_addr(addr);
		unsigned cache_index = (unsigned)-1;
		unsigned woc_cache_index = (unsigned)-1;
		enum distill_cache_request_status status = m_tag_array->probe(addr, cache_index, woc_cache_index);




/*
		if(status == LOC_HIT){ // Same as normal cache line
			// Cache line evicted from L1 --> Write here
			cache_block_t evicted(m_config.get_line_sz()), woc_evicted(m_config.get_line_sz());
			bool wb= false;
			m_tag_array->access(addr,time,cache_index, wb, evicted, woc_evicted); // update LRU state

			if ( evict ) { // Global write
				// Don't worry about WOC, just write back global

				 if ( m_miss_queue.size() >= m_config.m_miss_queue_size )
					 return RESERVATION_FAIL; // cannot handle request this cycle

				 // generate a write through
				 cache_block_t &block = m_tag_array->get_block(cache_index);
				 assert( block.m_status != MODIFIED ); // fails if block was allocated by a ld.local and now accessed by st.global

				 mf->set_all_valid_words(); // All words in the LOC are valid
				 m_miss_queue.push_back(mf);
				 mf->set_status(m_miss_queue_status,time);
				 events.push_back(WRITE_REQUEST_SENT);

				 // invalidate block
				 block.m_status = INVALID;
			 } else { // Global/Local read or Local write
				 m_stats.hit_event( addr, evicted, cache_index, mf, status );

				 if ( wr ) {
					 assert( type == LOCAL_ACC_W ||  type == L1_WRBK_ACC );
					 // treated as write back...
					 cache_block_t &block = m_tag_array->get_block(cache_index);
					 block.m_status = MODIFIED;
				 }
				 block.m_footprint |= mf->get_footprint();	// Mark used words??
				 mf->set_all_valid_words();
			 }
			 return HIT;

		}else if(status == WOC_HIT){ // Not in LOC, but word is in the WOC
			cache_block_t evicted(m_config.get_line_sz()), woc_evicted(m_config.get_line_sz());
			bool wb= false;
			m_tag_array->access(addr,time,cache_index, wb, evicted, woc_evicted); // update LRU state

			if(evict){ // Global write to WOC, evict
				m_tag_array->evict_words(addr, woc_evicted); // Evict GLOBAL words from WOC. Shouldn't be dirty
				mf->set_all_valid_words();
				m_miss_queue.push_back(mf);
				mf->set_status(m_miss_queue_status, time);
				events.push_back(WRITE_REQUEST_SENT);

			}else{ // Global/local read or local write
				 if(wr){
					 assert( type == LOCAL_ACC_W ||  type == L1_WRBK_ACC );
					 woc_cache_block_t &block = m_tag_array->get_block(woc_evicted);
					 block.WOC[m_config.word_id(addr)].state = MODIFIED;
				 }
				 mf->set_all_valid_words(); // Unsure about this line.
			}

		}else if(status != DISTILL_RESERVATION_FAIL){ // HOLE or LINE miss
			cache_block_t evicted(m_config.get_line_sz()), woc_evicted(m_config.get_line_sz());
			bool wb = false;
			m_tag_array->access(addr,time,cache_index, wb, evicted, woc_evicted); // update LRU state

			if(wr){
				if ( m_miss_queue.size() >= m_config.m_miss_queue_size )
					return RESERVATION_FAIL; // cannot handle request this cycle

				mf->set_all_valid_words();
				m_miss_queue.push_back(mf);
				mf->set_status(m_miss_queue_status,time);
				events.push_back(WRITE_REQUEST_SENT);
				return MISS;
			}else{ // Line/word miss but have some words in the WOC. Copy in data.
				if(status == HOLE_MISS){
					m_tag_array->evict_words(addr, woc_evicted); // Equivalent to copying the data back into the reserved line
					// Do something to set bits in allocated line?
				}

				if ( mshr_hit && mshr_avail ) {
				    //status = m_tag_array->access(addr,time,cache_index,wb,evicted, woc_evicted);
				    mf->set_all_valid_words();
				    m_mshrs.add(block_addr,mf);
				    do_miss = true;
				} else if ( !mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size) ) {
				    //status = m_tag_array->access(addr,time,cache_index,wb,evicted, woc_evicted);
				    mf->set_all_valid_words();
				    m_mshrs.add(block_addr,mf);
				    m_extra_mf_fields[mf] = extra_mf_fields(block_addr,cache_index, mf->get_data_size());
				    mf->set_data_size( m_config.get_line_sz() );
				    m_miss_queue.push_back(mf);
				    mf->set_status(m_miss_queue_status,time);
				    events.push_back(READ_REQUEST_SENT);
				    do_miss = true;
				}
				if( wb ) {
					mem_fetch *wb_blk;
					if(status == HOLE_MISS){	// Write back
						assert( (woc_cache_index != (unsigned)-1) && do_miss);
						wb_blk = m_memfetch_creator->alloc(woc_evicted.m_block_addr, L1_WRBK_ACC, m_config.get_line_sz(),true);
						wb_blk->set_all_valid_words();


						events.push_back(WRITE_BACK_REQUEST_SENT);
						m_miss_queue.push_back(wb_blk);
						wb_blk->set_status(m_miss_queue_status,time);
					}
					assert(do_miss);
					wb_blk = m_memfetch_creator->alloc(evicted.m_block_addr,L1_WRBK_ACC,m_config.get_line_sz(),true);
					wb_blk->set_all_valid_words();
					events.push_back(WRITE_BACK_REQUEST_SENT);
					m_miss_queue.push_back(wb_blk);
					wb_blk->set_status(m_miss_queue_status,time);
				}
				if( do_miss )
					return MISS;

			}




		}else{
			assert(status == DISTILL_RESERVATION_FAIL);
		}

*/

 	 	 if(status == LOC_HIT){
			    // Cache line evicted from L1 --> Write here
	       		cache_block_t evicted(m_config.get_line_sz()), woc_evicted(m_config.get_line_sz());
	       		bool wb= false;
	       		m_tag_array->access(addr,time,cache_index, wb, evicted, woc_evicted); // update LRU state

				if ( wr ) {
					//assert( type == LOCAL_ACC_W || type == L1_WRBK_ACC ); // Can also be global in the L2?
					// treated as write back...
					woc_cache_block_t &block = m_tag_array->get_block(cache_index);
					block.m_status = MODIFIED;
					block.m_footprint |= mf->get_footprint();	// Mark used words
				}

				mf->set_all_valid_words();
				return HIT;

		}else if(status == WOC_HIT){
			cache_block_t evicted(m_config.get_line_sz()), woc_evicted(m_config.get_line_sz());
	       		bool wb= false;	
			m_tag_array->access(addr,time,cache_index, wb, evicted, woc_evicted); // update LRU state
			woc_cache_block_t &block = m_tag_array->get_block(cache_index);

			mf->reset_valid_words();
			for(unsigned j=0; j<words_per_block; j++){
				if(m_config.tag(addr) == block.WOC[j].tag && block.WOC[j].state != INVALID){
					mf->set_valid_words(block.WOC[j].word_id, 1);
				}
			} 
			//mf->set_valid_words(block.m_valid_words);
			if(wr)
				block.m_status = MODIFIED;
			
			return HIT;
			
		}else if(status != DISTILL_RESERVATION_FAIL){
			if ( wr ) {
				if ( m_miss_queue.size() >= m_config.m_miss_queue_size )
					return RESERVATION_FAIL; // cannot handle request this cycle
				// on miss, generate write through (no write buffering -- too many threads for that)
				mf->set_all_valid_words();
				m_miss_queue.push_back(mf); 
				mf->set_status(m_miss_queue_status,time); 
				events.push_back(WRITE_REQUEST_SENT);
				return MISS;			
			} else {
		        if ( (m_miss_queue.size()+1) >= m_config.m_miss_queue_size )
                    return RESERVATION_FAIL; // cannot handle request this cycle (might need to generate two requests)
				bool do_miss = false;
				bool wb = false;
				cache_block_t evicted(m_config.get_line_sz());
				woc_cache_block_t woc_evicted(m_config.get_line_sz());

				bool mshr_hit = m_mshrs.probe(block_addr);
				bool mshr_avail = !m_mshrs.full(block_addr);

				status = m_tag_array->access(addr,time,cache_index,wb,evicted, woc_evicted);
				if ( mshr_hit && mshr_avail ) {
				    mf->set_all_valid_words();
				    m_mshrs.add(block_addr,mf);
				    do_miss = true;
				} else if ( !mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size) ) {
				    mf->set_all_valid_words();
				    m_mshrs.add(block_addr,mf);
				    m_extra_mf_fields[mf] = extra_mf_fields(block_addr,cache_index, mf->get_data_size());
				    mf->set_data_size( m_config.get_line_sz() );
				    m_miss_queue.push_back(mf);
				    mf->set_status(m_miss_queue_status,time); 
				    events.push_back(READ_REQUEST_SENT);
				    do_miss = true;
				}
				if( wb ) {
					mem_fetch *wb_blk;
					if(status == HOLE_MISS){	// Write back 
						assert( (woc_cache_index != (unsigned)-1) && do_miss);
						wb_blk = m_memfetch_creator->alloc(woc_evicted.m_block_addr, L1_WRBK_ACC, m_config.get_line_sz(),true);
						//wb_blk->set_valid_words(m_tag_array->get_block(woc_evicted).m_valid_words); // Setting valid words to be written to memory
						wb_blk->set_valid_words(woc_evicted.m_valid_words);
						//wb_blk->set_all_valid_words();
						events.push_back(WRITE_BACK_REQUEST_SENT);
						m_miss_queue.push_back(wb_blk);
						wb_blk->set_status(m_miss_queue_status,time); 
					}
					assert(do_miss);
					wb_blk = m_memfetch_creator->alloc(evicted.m_block_addr,L1_WRBK_ACC,m_config.get_line_sz(),true);
					wb_blk->set_all_valid_words();
					events.push_back(WRITE_BACK_REQUEST_SENT);
					m_miss_queue.push_back(wb_blk);
					wb_blk->set_status(m_miss_queue_status,time); 
				}
				if( do_miss ) 
					return MISS;
	
			}
		}
		return RESERVATION_FAIL;

	}


// interface for response from lower memory level (model bandwidth restictions in caller)
    virtual void fill( mem_fetch *mf, unsigned time )
    {
        extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
        m_stats.fill_event( mf, time, e->second.m_block_addr, e->second.m_cache_index );
        assert( e != m_extra_mf_fields.end() );
        assert( e->second.m_valid );

        m_tag_array->get_block( e->second.m_cache_index ).set_valid_words(mf->get_valid_words()); // TODO: Tayler - For Distillation Cache
        mf->set_data_size( e->second.m_data_size );

        if ( m_config.m_alloc_policy == ON_MISS )
            m_tag_array->fill(e->second.m_cache_index,time);
        else if ( m_config.m_alloc_policy == ON_FILL ){
            m_tag_array->fill(e->second.m_block_addr,time);
        } else abort();
/*
        cache_block_t &block = m_tag_array->get_block(e->second.m_cache_index);
	for(unsigned i=0; i<words_per_block; i++)
		block.m_footprint.set(mf->get_footprint(i));
*/
        bool has_atomic = false;
        m_mshrs.mark_ready(e->second.m_block_addr, has_atomic);
        m_extra_mf_fields.erase(mf);
    }

private:
    distill_tag_array* m_tag_array;
};

class sector_cache : public evict_on_write_cache {
public:
    sector_cache( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport, 
                mem_fetch_allocator *mfcreator, enum mem_fetch_status status, bool enable_access_dump )
    : evict_on_write_cache(name,config,core_id,type_id,memport,mfcreator,status, enable_access_dump) {
        evict_on_write_cache::m_tag_array = m_tag_array = new sector_tag_array( config, core_id, type_id, enable_access_dump );
    }

    virtual ~sector_cache() {
        delete m_tag_array;
    }

	virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events );
	virtual void fill( mem_fetch *mf, unsigned time );


private:
    sector_tag_array* m_tag_array;
};

#endif
