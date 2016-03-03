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

#ifndef GPU_CACHE_H
#define GPU_CACHE_H

#include <stdio.h>
#include <stdlib.h>
#include <set>
#include "gpu-misc.h"
#include "mem_fetch.h"
#include "abstract_hardware_model.h"
#include "tr1_hash_map.h"

#define ODD_HASH_MULTIPLE 7

class shader_core_stats;
class shader_page_reordering_tlb;

enum cache_block_state {
    INVALID,
    RESERVED,
    VALID,
    MODIFIED
};

enum cache_request_status {
    HIT = 0,
    HIT_RESERVED,
    MISS,
    RESERVATION_FAIL,
    FAIL_PROTECTION,
    NUM_CACHE_REQUEST_STATUS
};

enum cache_event {
    WRITE_BACK_REQUEST_SENT,
    READ_REQUEST_SENT,
    WRITE_REQUEST_SENT
};

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

struct cache_block_stats {
   void init( unsigned line_size ) {
      already_flushed = true;
      m_byte_mask.clear();
      for( unsigned i=0; i < line_size ; i++ )
         m_byte_mask.push_back( 0 );
   }

   std::vector<int> m_byte_mask;
   bool already_flushed;
};

struct cache_block_t {
    cache_block_t( unsigned line_size )
    {
        m_tag=0;
        m_block_addr=0;
        m_alloc_time=0;
        m_fill_time=0;
        m_last_access_time=0;
        m_status=INVALID;
        m_stats.init( line_size );
    }

    virtual ~cache_block_t() {}

    virtual void allocate( new_addr_type tag, new_addr_type block_addr, unsigned time )
    {
        m_tag=tag;
        m_block_addr=block_addr;
        m_alloc_time=time;
        m_last_access_time=time;
        m_fill_time=0;
        m_status=RESERVED;
    }

    virtual void fill( unsigned time )
    {
        assert( m_status == RESERVED );
        m_status=VALID;
        m_fill_time=time;
    }

    void print( FILE *fp ) const
    {
        const char *status = "";
        switch(m_status) {
        case INVALID:  status = "INVALID "; break;
        case MODIFIED: status = "MODIFIED"; break;
        case RESERVED: status = "RESERVED"; break;
        case VALID:    status = "VALID   "; break;
        }
        fprintf(fp,"block address=0x%08llx, status=%s, last access time = %u\n",
                m_block_addr, status, m_last_access_time );
    }

    new_addr_type    m_tag;
    new_addr_type    m_block_addr;
    unsigned         m_alloc_time;
    unsigned         m_last_access_time;
    unsigned         m_fill_time;
    cache_block_state    m_status;
    unsigned    m_line_sz;

    cache_block_stats m_stats;
};

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
#define NUM_SECTORED 8
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
//        m_status=RESERVED;
		if(m_status == INVALID)
			m_status = RESERVED;

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

struct virtual_policy_cache_block_t : public cache_block_t {
    virtual_policy_cache_block_t( unsigned line_size )
        : cache_block_t( line_size ) { }
    virtual ~virtual_policy_cache_block_t() {}
    virtual void fill( unsigned time )
    {
        // tgrogers - changing the logic here for the purposes of our vm remapping system that may invalidate lines that
        //            have not yet been filled.  In which case, lets just ignore these fills...
        //assert( m_status == RESERVED );
        if ( m_status == RESERVED ) {
            m_status=VALID;
            m_fill_time=time;
        }
    }

    vm_page_mapping_payload m_policy;
};

struct priority_cache_block : public cache_block_t {
    priority_cache_block( unsigned line_size )
        : cache_block_t( line_size ), m_cycles_protected( 0 ), m_accesses_protected( 0 ), m_source_pc( 0 ) {}
    virtual ~priority_cache_block(){}

    virtual void allocate( new_addr_type tag, new_addr_type block_addr, unsigned time ) {
        cache_block_t::allocate( tag, block_addr, time );
        // TODO tgrogers - the line protection should happen here...
    }

    unsigned m_cycles_protected;
    unsigned m_accesses_protected;
    std::map< unsigned, unsigned > m_warps_protecting_line;
    new_addr_type m_source_pc;
};

struct victim_cache_block : public cache_block_t {
    victim_cache_block( unsigned line_size )
        : cache_block_t( line_size ), m_source_pc( 0 ) {}
    virtual ~victim_cache_block(){}

    new_addr_type m_source_pc;
};

enum replacement_policy_t {
    LRU,
    FIFO,
    DIVERGENT_MRU,
    STATIC_LOAD_PREDICTION,
    START_RRIP,
    SRRIP = START_RRIP,
    BRRIP,
    END_RRIP = BRRIP,
    BELADY_OPTIMAL,
    PROTECT_ALL_PENDING_DATA,
};

struct rrip_cache_block : public cache_block_t {
    static const unsigned M = 5;
    static const unsigned EVICT_RRPV = ( 1 << M ) - 1;
    rrip_cache_block( unsigned line_size, replacement_policy_t policy, float brrip_long_chance )
        : cache_block_t( line_size ), m_rrpv( ( 1 << M ) - 2 ), m_replacement_policy( policy )
        , m_brrip_prob_insert_long( brrip_long_chance ){
        assert( m_brrip_prob_insert_long >= 0.0f && m_brrip_prob_insert_long <= 100.0f );
    }

    virtual void allocate( new_addr_type tag, new_addr_type block_addr, unsigned time ) {
        if ( m_replacement_policy == SRRIP ) {
            m_rrpv = ( 1 << M ) - 2;
        } else if ( m_replacement_policy == BRRIP ) {
            // m_brrip_prob_insert_long is a pecentage, times 100 so it is accurate to 2 decimal places
            if ( (float)( rand() % 10000 ) < ( m_brrip_prob_insert_long * 100 ) ) {
                m_rrpv = ( 1 << M ) - 2;
            } else {
                m_rrpv = ( 1 << M ) - 1;
            }
        }

        cache_block_t::allocate( tag, block_addr, time );
    }

    unsigned m_rrpv;

    const replacement_policy_t m_replacement_policy;
    const float m_brrip_prob_insert_long;
};

enum cache_type {
    CACHE_NORMAL,
    CACHE_SECTORED,
    CACHE_VIRTUAL_POLICY_MANAGED,
    CACHE_HIGH_LOCALITY_PROTECTED
};

enum rrip_hit_policy_t {
    HP_RRIP_HIT_POLICY = 0,
    FP_RRIP_HIT_POLICY
};

enum write_policy_t {
    READ_ONLY,
    WRITE_BACK,
    WRITE_THROUGH
};

enum allocation_policy_t {
    ON_MISS,
    ON_FILL
};

enum mshr_config_t {
    TEX_FIFO,
    ASSOC // normal cache 
};


enum mem_div_status {
   MEM_DIV_INVALID,
   MEM_DIV_NEVERUSED,
   MEM_DIV_USEDATARRIVAL,
   MEM_DIV_USEDLATER
};


#define NUM_MEM_DIV_BINS	8
struct div_map_t {
    div_map_t(): status(MEM_DIV_INVALID),last_access_cycle(0), last_pc((address_type)-1) ,cycle_loaded(0), num_access(0) {}

    enum mem_div_status  status;
    unsigned long long  last_access_cycle;
    address_type        last_pc;
    address_type	cycle_loaded;
    unsigned		num_access;	
};

    
typedef struct _mem_div_history{
	_mem_div_history() : total_bytes_read(0), bytes_used_now(0), bytes_event_used(0), bytes_never_used(0) {for(unsigned i=0; i<NUM_MEM_DIV_BINS; i++){ bytes_used_in_cache[i]=0; bytes_event_used_in_cache[i]=0;}}
	unsigned total_bytes_read;
	unsigned bytes_used_now; // Cold start
	unsigned bytes_event_used;	
	unsigned bytes_never_used;

	unsigned bytes_used_in_cache[NUM_MEM_DIV_BINS]; // Fraction of bytes used per lifetime of each cache line
	unsigned bytes_event_used_in_cache[NUM_MEM_DIV_BINS];
}mem_div_history;


class cache_config {
public:
    cache_config() 
    { 
        m_valid = false; 
        m_disabled = false;
        m_config_string = NULL; // set by option parser
    }
    virtual void init( const char* config_str = NULL )
    {
        if ( !config_str ) config_str = m_config_string;
        assert( config_str );
        char rp, wp, ap, mshr_type, type;
        int ntok = sscanf(config_str,"%c,%u:%u:%u:%c:%c:%c,%c:%u:%u,%u:%u:%u:%f:%u",
                        &type, &m_nset, &m_line_sz, &m_assoc, &rp, &wp, &ap,
                        &mshr_type,&m_mshr_entries,&m_mshr_max_merge,&m_miss_queue_size,
                        &m_result_fifo_entries, &mrrip_hit_update_pol, &mbrrip_chance_insert_long,
                        &enable_cache_access_dump);
        if ( ntok < 15 ) {
            if ( !strcmp(config_str,"none") ) {
                m_disabled = true;
                return;
            }
            exit_parse_error();
        }
        switch( type ) {
        case 'N': m_cache_type = CACHE_NORMAL; break;
        case 'S': m_cache_type = CACHE_SECTORED; break;
        case 'V': m_cache_type = CACHE_VIRTUAL_POLICY_MANAGED; break;
        case 'L': m_cache_type = CACHE_HIGH_LOCALITY_PROTECTED; break;
        default: exit_parse_error();
        }
        switch (rp) {
        case 'L': m_replacement_policy = LRU; break;
        case 'F': m_replacement_policy = FIFO; break;
        case 'D': m_replacement_policy = DIVERGENT_MRU; break;
        case 'S': m_replacement_policy = STATIC_LOAD_PREDICTION; break;
        case 'R': m_replacement_policy = SRRIP; break;
        case 'B': m_replacement_policy = BRRIP; break;
        case 'O': m_replacement_policy = BELADY_OPTIMAL; break;
        case 'P': m_replacement_policy = PROTECT_ALL_PENDING_DATA; break;
        default: exit_parse_error();
        }
        switch (wp) {
        case 'R': m_write_policy = READ_ONLY; break;
        case 'B': m_write_policy = WRITE_BACK; break;
        case 'T': m_write_policy = WRITE_THROUGH; break;
        default: exit_parse_error();
        }
        switch (ap) {
        case 'm': m_alloc_policy = ON_MISS; break;
        case 'f': m_alloc_policy = ON_FILL; break;
        default: exit_parse_error();
        }
        switch (mshr_type) {
        case 'F': m_mshr_type = TEX_FIFO; break;
        case 'A': m_mshr_type = ASSOC; break;
        default: exit_parse_error();
        }
        m_line_sz_log2 = LOGB2(m_line_sz);
        m_nset_log2 = LOGB2(m_nset);
        m_valid = true;
    }
    bool disabled() const { return m_disabled;}
    unsigned get_line_sz() const
    {
        assert( m_valid );
        return m_line_sz;
    }
    unsigned get_num_lines() const
    {
        assert( m_valid );
        return m_nset * m_assoc;
    }

    void print( FILE *fp ) const
    {
        fprintf( fp, "Size = %d B (%d Set x %d-way x %d byte line)\n", 
                 m_line_sz * m_nset * m_assoc,
                 m_nset, m_assoc, m_line_sz );
    }

    unsigned set_index( new_addr_type addr ) const;

    new_addr_type tag( new_addr_type addr ) const
    {
        return addr >> (m_line_sz_log2+m_nset_log2);
    }
    new_addr_type block_addr( new_addr_type addr ) const
    {
        return addr & ~(m_line_sz-1);
    }
    char get_replacement_policy() const
    {
        return m_replacement_policy;
    }
    unsigned get_num_sets() const {
       return m_nset;
    }

    unsigned word_id( new_addr_type addr ) const{
       	return addr >> LOGB2(WORD_SIZE) & ((WORD_SIZE/2)-1);
    }

    enum cache_type get_cache_type() const {
        return m_cache_type;
    }

    bool is_rrip_policy() const {
        return m_replacement_policy >= START_RRIP && m_replacement_policy <= END_RRIP;
    }

    bool is_access_dump_enabled() const { return enable_cache_access_dump == 1; }

    char *m_config_string;

protected:
    unsigned set_index_hashed( new_addr_type addr ) const;

    void exit_parse_error()
    {
        printf("GPGPU-Sim uArch: cache configuration parsing error (%s)\n", m_config_string );
        abort();
    }

    bool m_valid;
    bool m_disabled;
    unsigned m_line_sz;
    unsigned m_line_sz_log2;
    unsigned m_nset;
    unsigned m_nset_log2;
    unsigned m_assoc;


    enum cache_type m_cache_type; // 'L' = LRU, 'F' = FIFO
    enum replacement_policy_t m_replacement_policy; // 'L' = LRU, 'F' = FIFO
    enum write_policy_t m_write_policy;             // 'T' = write through, 'B' = write back, 'R' = read only
    enum allocation_policy_t m_alloc_policy;        // 'm' = allocate on miss, 'f' = allocate on fill
    enum mshr_config_t m_mshr_type;

    union {
        unsigned m_mshr_entries;
        unsigned m_fragment_fifo_entries;
    };
    union {
        unsigned m_mshr_max_merge;
        unsigned m_request_fifo_entries;
    };
    union {
        unsigned m_miss_queue_size;
        unsigned m_rob_entries;
    };
    unsigned m_result_fifo_entries;

    unsigned mrrip_hit_update_pol;
    float mbrrip_chance_insert_long;

    unsigned enable_cache_access_dump;

    friend class base_tag_array;
    friend class tag_array;
    friend class evict_on_write_cache;
    friend class virtual_policy_cache;
    friend class priority_tag_array;
    friend class high_locality_protected_cache;
    friend class tag_based_cache_t;
    friend class read_only_cache;
    friend class tex_cache;
    friend class data_cache;
    friend class mem_div_cache;

    friend class distill_cache;
    friend class distill_tag_array;
    friend class rrip_tag_array;
    friend class cache_stats;
    friend class sector_cache;
    friend class sector_tag_array;
};

enum high_locality_cache_type {
    STATIC_HIGH_LOCALITY_CACHE,
    DYNAMIC_HIGH_LOCALITY_CACHE
};

enum victim_cache_scope {
    ALL_WARPS_VICTIM_CACHE_SCOPE,
    ONE_WARP_VICTIM_CACHE_SCOPE
};

class high_locality_cache_config {
public:
    high_locality_cache_config() : m_type( STATIC_HIGH_LOCALITY_CACHE ), m_num_victim_entries( 100 ),
        m_victim_cache_scope( ALL_WARPS_VICTIM_CACHE_SCOPE ), m_weight_to_protect( 10 ), m_so_factor( 32 ) {}

    virtual void init( const char* config_str = NULL );

    bool does_line_have_high_locality( const char* file, unsigned line ) const {
        return m_highlocality_source_lines.find( file ) != m_highlocality_source_lines.end()
                && m_highlocality_source_lines.find( file )->second.find( line ) != m_highlocality_source_lines.find( file )->second.end();
    }

    char *m_config_string;

    high_locality_cache_type get_type() const { return m_type; }
    unsigned get_num_victim_entries() const { return m_num_victim_entries; }
    victim_cache_scope get_victim_cache_scope() const { return victim_cache_scope( m_victim_cache_scope ); }
    unsigned get_weight_to_protect() const { return m_weight_to_protect; }
    unsigned get_so_factor() const { return m_so_factor; }
private:
    std::map< std::string, std::set< unsigned > > m_highlocality_source_lines;
    high_locality_cache_type m_type;
    unsigned m_num_victim_entries;
    unsigned m_victim_cache_scope;
    unsigned m_weight_to_protect;
    unsigned m_so_factor;
};


class tlb_config : public cache_config {
public:
    tlb_config() : m_translation_config( NO_TRANSLATION ) {}

    virtual void init( const char* config_str = NULL );

    translation_config get_translation_config() const { return m_translation_config; }
    virtual_page_type get_virtual_page_type() const { return m_virtual_page_type; }

private:
   translation_config m_translation_config;
   virtual_page_type m_virtual_page_type;
};

class base_tag_array;

class access_stream_entry {
public:
    access_stream_entry( new_addr_type n_addr,
                         enum mem_access_type n_mem_space_type,
                         cache_request_status n_access_status,
                         const mem_fetch* mf )
        : m_addr( n_addr ),
          m_mem_space_type( n_mem_space_type ),
          m_access_status( n_access_status ),
          m_warp_id( mf ? mf->get_wid() : 0 ),
          m_access_mask( mf ? mf->get_access_warp_mask() : active_mask_t() ),
          m_num_threads_touching( mf ? mf->get_access_warp_mask().count() : 0 ),
          m_pc( mf ? mf->get_pc() : 0 ) {}

    access_stream_entry( const char* str ) {
        char rw = 'u', type = 'u', status = 'u';
        m_mem_space_type = NUM_MEM_ACCESS_TYPE;
        const unsigned NUM_TOKENS = 7;
        const unsigned ntok = sscanf( str, "%llx:%c:%c:%c:%u:%zu:%llu", &m_addr, &rw, &type, &status,
                                      &m_warp_id, &m_num_threads_touching, &m_pc );
        if ( ntok != NUM_TOKENS ) {
            char flush;
            assert( sscanf( str, "%c", &flush ) == 1 );
            if ( 'F' == flush ) {
                m_addr = CACHE_FLUSH_SIGNATURE;
                m_warp_id = (unsigned)-1;
                return;
            } else {
                fprintf( stderr, "Error - Unknown Entry\n" );
                abort();
            }
        } else {
            assert( ntok == NUM_TOKENS );
        }

        if ( rw == 'r' ) {
                if ( type == 'g' ) {
                    m_mem_space_type = GLOBAL_ACC_R;
                } else if ( type == 'l' ) {
                    m_mem_space_type = LOCAL_ACC_R;
                }
        } else if ( rw == 'w' ) {
            if ( type == 'g' ) {
                m_mem_space_type = GLOBAL_ACC_W;
            } else if ( type == 'l' ) {
                m_mem_space_type = LOCAL_ACC_W;
            }
        }
        assert( NUM_MEM_ACCESS_TYPE != m_mem_space_type );
        for ( unsigned i = 0; i < NUM_CACHE_REQUEST_STATUS; ++i ) {
            if ( status == m_status_to_char_map[ i ] ) {
                m_access_status = cache_request_status( i );
                break;
            }
        }
    }

    void set_addr( new_addr_type n_addr ){ m_addr = n_addr; }
    void set_access_type( mem_access_type n_mem_space_type ) {
        m_mem_space_type = n_mem_space_type;
    }
    void set_request_status( cache_request_status n_access_status ) {
        m_access_status = n_access_status;
    }

    void print( FILE* output ) const  {
        print( output, "\n" );
    }

    void print( FILE* output, const char* delimiter ) const {
        if ( !is_cache_flush() ) {
            char entry = 'u';
            char rw = 'u';
            char cache_status = 'u';
            switch ( m_mem_space_type ) {
                case GLOBAL_ACC_W: {
                    entry = 'g';
                    rw = 'w';
                } break;
                case GLOBAL_ACC_R: {
                    entry = 'g';
                    rw = 'r';
                } break;
                case LOCAL_ACC_W: {
                    entry = 'l';
                    rw = 'w';
                } break;
                case LOCAL_ACC_R: {
                    entry = 'l';
                    rw = 'w';
                } break;
                default:
                    fprintf( stderr,
                            "Error - currently only local and global memory access can be logged\n" );
                    abort();
           }
           assert( m_access_status >= 0 && m_access_status < NUM_CACHE_REQUEST_STATUS );
           cache_status = m_status_to_char_map[ m_access_status ];
           fprintf( output, "%llx:%c:%c:%c:%u:%zu:%llu", m_addr, rw, entry, cache_status,
                    m_warp_id, m_num_threads_touching, m_pc );
        } else {
          fprintf( output, "F" );
        }
        fprintf( output, delimiter);
    }

    bool is_cache_flush() const { return CACHE_FLUSH_SIGNATURE == m_addr; }

    static const new_addr_type CACHE_FLUSH_SIGNATURE = 0xDEADBEEFFEEBDAED;

    new_addr_type get_addr() const { return m_addr; }
    mem_access_type get_mem_space_type() const { return m_mem_space_type; }
    unsigned get_wid() const { return m_warp_id; }
    new_addr_type get_pc() const { return m_pc; }
    new_addr_type get_num_lanes_touching() const { return m_num_threads_touching; }
private:
    static const char m_status_to_char_map[ NUM_CACHE_REQUEST_STATUS ];// = {'h','r','m','f','p'};
    new_addr_type m_addr;
    mem_access_type m_mem_space_type;
    cache_request_status m_access_status;
    unsigned m_warp_id;
    const active_mask_t m_access_mask;
    size_t m_num_threads_touching; // Normally this is just m_access_mask.count()
                                     // but in the SA sim it is read form the text file
                                     // just cause I don't want to print out the whole access mask
    new_addr_type m_pc;
};

struct tag_array_stats {
   tag_array_stats( const base_tag_array* ta ) : m_tag_array ( ta ) {}

   inline void init();
   inline void set_byte_mask(unsigned block_index, unsigned index, int val);
   inline int get_byte_mask(unsigned block_index, unsigned index);
   inline void clear_byte_mask(unsigned block_index);
   inline void set_line_first_use_status(unsigned idx, bool val);
   inline bool line_first_use_status(unsigned idx) const;
   inline new_addr_type evicted_blk_addr() { return m_evicted_block_addr; }

   const base_tag_array* m_tag_array;
   new_addr_type m_evicted_block_addr;
   std::list< access_stream_entry > m_access_stream;
};

// ctor is protected here.  This class is not meant to be instantiated, only derived from
class base_tag_array {
public:
    virtual ~base_tag_array();

    enum cache_request_status probe( new_addr_type addr, unsigned &idx, bool &all_reserved ) const;
    virtual enum cache_request_status probe( new_addr_type addr, unsigned &idx ) const;
    enum cache_request_status access( new_addr_type addr, unsigned time, unsigned &idx, const mem_fetch* mf );
    virtual enum cache_request_status access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted, const mem_fetch* mf );

    virtual void fill( new_addr_type addr, unsigned time );
    void fill( unsigned idx, unsigned time );

    unsigned size() const { return m_config.get_num_lines();}
    cache_block_t &get_block(unsigned idx) const {
        return *m_lines[idx];
    }

    void flush(); // flash invalidate all entries
    void new_window();

    void print( FILE *stream, unsigned &total_access, unsigned &total_misses ) const;
    float windowed_miss_rate( bool minus_pending_hit ) const;
    void print_set( new_addr_type addr, FILE *stream ) const;

    void print_tag_info() const;

    tag_array_stats& get_stats() const { return m_stats; }
    const cache_config& get_config() const { return m_config; }

protected:
    base_tag_array( const cache_config &config, int core_id, int type_id, bool m_enable_access_stream_dump );

    const cache_config &m_config;

    std::vector< cache_block_t * > m_lines; /* nbanks x nset x assoc lines in total */

    unsigned m_access;
    unsigned m_miss;
    unsigned m_pending_hit; // number of cache miss that hit a line that is allocated but not filled

    // performance counters for calculating the amount of misses within a time window
    unsigned m_prev_snapshot_access;
    unsigned m_prev_snapshot_miss;
    unsigned m_prev_snapshot_pending_hit;

    int m_core_id; // which shader core is using this
    int m_type_id; // what kind of cache is this (normal, texture, constant)

    // We should not make any functionality decisions based on the stats, making it mutable for this purpose
    mutable tag_array_stats m_stats;
    bool m_enable_access_stream_dump;
};

class tag_array : public base_tag_array {
public:
    tag_array( const cache_config &config, int core_id, int type_id, bool enable_access_stream_dump ) : base_tag_array( config, core_id, type_id, enable_access_stream_dump ) {
        for ( unsigned i = 0; i < config.get_num_lines(); ++i ) {
            m_lines[ i ] = new cache_block_t( config.get_line_sz() );
        }
    }

    virtual ~tag_array() {
        for ( unsigned i = 0; i < m_config.get_num_lines(); ++i ) {
            delete m_lines[ i ];
        }
    }
};

class rrip_tag_array : public base_tag_array {
public:
    rrip_tag_array( const cache_config &config, int core_id, int type_id, bool enable_access_stream_dump ) : base_tag_array( config, core_id, type_id, enable_access_stream_dump ) {
        m_lines.resize( config.get_num_lines(), NULL );
        for ( unsigned i = 0; i < config.get_num_lines(); ++i ) {
            base_tag_array::m_lines[ i ] = m_lines[ i ] =
                    new rrip_cache_block( config.get_line_sz(),
                            config.m_replacement_policy,
                            config.mbrrip_chance_insert_long );
        }
    }

    virtual ~rrip_tag_array() {
        for ( unsigned i = 0; i < m_config.get_num_lines(); ++i ) {
            delete m_lines[ i ];
        }
    }

    void increase_rrip( new_addr_type block_addr );
    void handle_hit( new_addr_type block_addr );

    virtual void fill( new_addr_type addr, unsigned time ) {
        assert( m_config.m_alloc_policy == ON_FILL );
        unsigned idx;
        enum cache_request_status status = probe(addr,idx);
        assert( status==MISS ||status == RESERVATION_FAIL );
        while ( status != MISS ) {
            increase_rrip( addr );
            status = probe( addr, idx );
        }
        base_tag_array::fill( addr, time );
    }

private:
    std::vector< rrip_cache_block * > m_lines; /* nbanks x nset x assoc lines in total */
};

inline void tag_array_stats::set_byte_mask(unsigned block_index, unsigned index, int val) {
   m_tag_array->get_block( block_index ).m_stats.m_byte_mask.at(index) = val;
}

inline int tag_array_stats::get_byte_mask(unsigned block_index, unsigned index){
   return m_tag_array->get_block( block_index ).m_stats.m_byte_mask.at(index);
}

inline void tag_array_stats::clear_byte_mask(unsigned block_index){
   m_tag_array->get_block( block_index ).m_stats.m_byte_mask.clear();
}

inline void tag_array_stats::set_line_first_use_status(unsigned idx, bool val){
   m_tag_array->get_block( idx ).m_stats.already_flushed = val;
}

inline bool tag_array_stats::line_first_use_status(unsigned idx) const {
   return m_tag_array->get_block( idx ).m_stats.already_flushed;
}

inline void tag_array_stats::init() {
   for ( unsigned i = 0; i < m_tag_array->get_config().get_num_lines(); i++ ) {
      m_tag_array->get_block( i ).m_stats.init(  m_tag_array->get_config().get_line_sz() );
   }
}

class cache_stats {
public:
    cache_stats( const cache_config& config, base_tag_array* const& ta )
       : m_config( config ), m_tag_array( ta ) {
        m_set_access_number = new unsigned[ m_config.get_num_sets() ];
        memset( m_set_access_number, 0x0, m_config.get_num_sets() * sizeof( unsigned ) );
        m_set_miss_number = new unsigned[ m_config.get_num_sets() ];
        memset( m_set_miss_number, 0x0, m_config.get_num_sets() * sizeof( unsigned ) );
    }

    virtual ~cache_stats() {
        delete[] m_set_access_number;
        delete[] m_set_miss_number;
    }

    unsigned get_never_used() const {
       return m_d_h.bytes_never_used;
    }
    unsigned get_used_now() const {
       return m_d_h.bytes_used_now;
    }
    unsigned get_event_used() const {
       return m_d_h.bytes_event_used;
    }

    unsigned get_used_cache_array(unsigned idx) const {
        assert(idx < NUM_MEM_DIV_BINS);
        return m_d_h.bytes_used_in_cache[idx];
    }

    unsigned get_event_used_cache_array(unsigned idx) const {
        assert(idx < NUM_MEM_DIV_BINS);
        return m_d_h.bytes_event_used_in_cache[idx];
    }

    void caclulate_useful_bytes_evicted(new_addr_type addr, cache_block_t evicted, unsigned idx, mem_fetch *mf, enum cache_request_status status ){
      if(mf->is_L2_access()) return;
      unsigned offset = 0;
      assert(mf != NULL); // Necessary?

      mem_access_byte_mask_t byte_mask = mf->get_byte_mask();
      if(m_config.get_line_sz() > MAX_MEMORY_ACCESS_SIZE){  // Temporary.. assumes 128 or 256 byte line sizes...
         offset = ( (addr - m_config.block_addr(addr)) < (m_config.get_line_sz() / 2) ? 0 : (m_config.get_line_sz() / 2));
      }

      if(status == HIT || status == HIT_RESERVED){ // Mark bytes as useful
         addr = m_config.block_addr(addr);
         for(unsigned i=offset; i<byte_mask.size()+offset; i++){
            if(byte_mask.test(i-offset)){
             //if(m_tag_array->get_byte_mask(idx, i) == 0){
               //    m_tag_array->set_byte_mask(idx, i, 2);
             //}
               m_tag_array->get_stats().set_byte_mask(idx, i, 1);
            }
         }
      }else if(status == MISS){  // Record cache line stats, mark bytes as immediately useful in new cache line
         addr = m_config.block_addr(addr);
         unsigned tot_bytes_used = 0;
      //unsigned tot_bytes_event_used = 0;
         for(unsigned i=0; i<m_config.get_line_sz(); i++){
            if(!m_tag_array->get_stats().line_first_use_status(idx)){
               if(m_tag_array->get_stats().get_byte_mask(idx, i) == 0)
                  m_d_h.bytes_never_used++;
               else if(m_tag_array->get_stats().get_byte_mask(idx, i) == 1){
                  tot_bytes_used++;
                  m_d_h.bytes_used_now++;
               }//else{ // == 2
            // tot_bytes_event_used++;
            // m_d_h.bytes_event_used++;
            //}
            }
            m_tag_array->get_stats().set_byte_mask(idx, i, 0);
            if(i >= offset && (i < byte_mask.size()+offset) && byte_mask.test(i-offset))
               m_tag_array->get_stats().set_byte_mask(idx, i-offset, 1);

         }
         unsigned index = ( (float)tot_bytes_used*NUM_MEM_DIV_BINS ) / (float)m_config.get_line_sz();
         if(index == NUM_MEM_DIV_BINS) index = NUM_MEM_DIV_BINS - 1;
         m_d_h.bytes_used_in_cache[index] += tot_bytes_used;

    /*
      index = ( (float)tot_bytes_event_used*NUM_MEM_DIV_BINS ) / (float)m_config.get_line_sz();
         if(index == NUM_MEM_DIV_BINS) index = NUM_MEM_DIV_BINS - 1;
      m_d_h.bytes_event_used_in_cache[index] += tot_bytes_event_used;
    */
         m_tag_array->get_stats().set_line_first_use_status(idx, false);
      } else { // Reservation fail or worse... Do nothing

      }
    }

    void final_eval_cache_lines(){
      // At the end of each kernel, evaluate the number of useful bytes in each cache line (Not counted yet on evict)
      unsigned tot_bytes_used = 0;
      unsigned tot_bytes_event_used = 0;
      for(unsigned idx=0; idx<m_config.get_num_lines(); idx++){
         if(!m_tag_array->get_stats().line_first_use_status(idx)){ // Not the first allocation of this block
            for(unsigned i=0; i<m_config.get_line_sz(); i++){
               if(m_tag_array->get_stats().get_byte_mask(idx, i) == 0)
                  m_d_h.bytes_never_used++;
               else if(m_tag_array->get_stats().get_byte_mask(idx, i) == 1){
                  tot_bytes_used++;
                  m_d_h.bytes_used_now++;
               }//else{ // == 2
    //   tot_bytes_event_used++;
    //   m_d_h.bytes_event_used++;
    //       }
               m_tag_array->get_stats().set_byte_mask(idx, i, 0);  // Clear the mask
            }
            unsigned index = ( (float)tot_bytes_used*NUM_MEM_DIV_BINS) / (float)m_config.get_line_sz();
            if(index == NUM_MEM_DIV_BINS ) index = NUM_MEM_DIV_BINS -1;
            m_d_h.bytes_used_in_cache[index] += tot_bytes_used;

    /*
       index = ( (float)tot_bytes_event_used*NUM_MEM_DIV_BINS ) / (float)m_config.get_line_sz();
          if(index == NUM_MEM_DIV_BINS) index = NUM_MEM_DIV_BINS - 1;
       m_d_h.bytes_event_used_in_cache[index] += tot_bytes_event_used;
    */
            tot_bytes_used = 0;
            tot_bytes_event_used = 0;
         }
         m_tag_array->get_stats().set_line_first_use_status(idx, true);
      }
    }

    void fill_event( const mem_fetch *mf, unsigned time, new_addr_type block_addr, unsigned cache_index ) {
      if ( m_config.m_alloc_policy == ON_FILL ){
         unsigned idx=0;
         m_tag_array->probe(block_addr, idx);
         set_evicted_block_addr( m_tag_array->get_block( idx ).m_block_addr );
         set_new_block_addr( block_addr );
      }
    }

    void access_event( new_addr_type addr ) {
        new_addr_type cache_line = m_config.block_addr(addr);
        if ( m_cache_lines_accessed.find( cache_line ) == m_cache_lines_accessed.end() ) {
            m_cache_lines_accessed[ cache_line ] = 1;
        } else {
            m_cache_lines_accessed[ cache_line ]++;
        }
        ++m_set_access_number[ m_config.set_index(addr) ];
    }

    void miss_event( new_addr_type addr, cache_block_t evicted, unsigned idx, mem_fetch *mf, enum cache_request_status status ) {
        m_set_miss_number[ m_config.set_index(addr) ]++;
        caclulate_useful_bytes_evicted(addr, evicted, idx, mf, status); // Calculate useful data on this cache access
    }

    void hit_event( new_addr_type addr, cache_block_t evicted, unsigned idx, mem_fetch *mf, enum cache_request_status status ) {
        caclulate_useful_bytes_evicted(addr, evicted, idx, mf, status); // Calculate useful data on this cache access
    }

    void set_evicted_block_addr( new_addr_type ev_blk){
       evicted_blk_addr = ev_blk;
    }

    new_addr_type get_evicted_block_addr(){
       return evicted_blk_addr;
    }

    void set_new_block_addr( new_addr_type nw_blk ){
       new_blk_addr = nw_blk;
    }

    new_addr_type get_new_block_addr(){
       return new_blk_addr;
    }

    unsigned get_set_access_histo( unsigned set_num ) const { return m_set_access_number[ set_num ]; }
    unsigned get_set_miss_histo( unsigned set_num ) const { return m_set_miss_number[ set_num ]; }

private:
    const cache_config& m_config;
    base_tag_array* const & m_tag_array;
    std::map< new_addr_type, unsigned > m_cache_lines_accessed;
    mem_div_history m_d_h;
    unsigned* m_set_access_number;
    unsigned* m_set_miss_number;

    new_addr_type evicted_blk_addr;
    new_addr_type new_blk_addr;
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

	enum sector_cache_request_status probe_sector( new_addr_type addr, unsigned &idx, bool &all_reserved) const;
	enum sector_cache_request_status access_sector( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted);
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

class virtual_policy_tag_array : public base_tag_array {
public:
    virtual_policy_tag_array( const cache_config &config, int core_id, int type_id, bool enable_access_stream_dump )
        : base_tag_array( config,core_id,type_id,enable_access_stream_dump ) {
        m_lines.resize( config.get_num_lines(), NULL );
        for ( unsigned i = 0; i < config.get_num_lines(); ++i ) {
            base_tag_array::m_lines[ i ] = m_lines[ i ] = new virtual_policy_cache_block_t( config.get_line_sz() );
        }
    }
    virtual ~virtual_policy_tag_array(){
        for ( unsigned i = 0; i < m_config.get_num_lines(); ++i ) {
            delete m_lines[ i ];
        }
    }

    virtual_policy_cache_block_t &get_block(unsigned idx) const { return *m_lines[ idx ]; }

private:
    std::vector< virtual_policy_cache_block_t * > m_lines;
};

extern unsigned PROTECTED_CYCLES;
extern unsigned PROTECTED_ACCESES;
enum protection_type {
    CYCLES_PROTECTION_TYPE = 0,
    ACCESSES_PROTECTION_TYPE
};
class priority_tag_array : public base_tag_array {
public:
    priority_tag_array( const cache_config &config, int core_id, int type_id, unsigned num_warps, unsigned so_factor, bool enable_access_stream_dump )
        : base_tag_array( config,core_id,type_id, enable_access_stream_dump ), m_num_protected_lines( 0 ), m_stall_override_factor( so_factor ) {
        m_lines.resize( config.get_num_lines(), NULL );
        for ( unsigned i = 0; i < config.get_num_lines(); ++i ) {
            base_tag_array::m_lines[ i ] = m_lines[ i ] = new priority_cache_block( config.get_line_sz() );
        }
        m_lines_protected_per_warp.resize( num_warps );
        for ( unsigned i = 0; i < num_warps; ++i ) {
            m_lines_protected_per_warp[ i ].first = i;
        }
    }

    virtual ~priority_tag_array(){
        for ( unsigned i = 0; i < m_config.get_num_lines(); ++i ) {
            delete m_lines[ i ];
        }
    }

    void cycle();

    void remove_protected_lines( unsigned warp_id ) {
        //printf( "removing protection for warp=%u\n", warp_id );
        for ( std::vector< priority_cache_block * >::const_iterator it = m_lines.begin(); it != m_lines.end(); ++it ) {
            if ( (*it)->m_warps_protecting_line.find( warp_id ) != (*it)->m_warps_protecting_line.end() ) {
                (*it)->m_warps_protecting_line.erase( warp_id );
            }
        }
    }

    void print_protected_info( FILE* fout ) {
        /*
        unsigned num_lines_protected = 0;
        if ( get_protection_type() == CYCLES_PROTECTION_TYPE ) {
            for ( std::vector< priority_cache_block * >::const_iterator it = m_lines.begin(); it != m_lines.end(); ++it ) {
                if ( ( !hack_use_warp_prot_list && (*it)->m_cycles_protected > 0 ) || ( hack_use_warp_prot_list && (*it)->m_warps_protecting_line.size() > 0 ) ) {
                    ++num_lines_protected;
                }
            }
        } else if ( get_protection_type() == ACCESSES_PROTECTION_TYPE ) {
            for ( std::vector< priority_cache_block * >::const_iterator it = m_lines.begin(); it != m_lines.end(); ++it ) {
                if ( (*it)->m_accesses_protected > 0 ) {
                    ++num_lines_protected;
                }
            }
        } else {
            abort();
        }

        fprintf( fout, "cache_protected_lines=%u\ncache_percent_lines_protected=%f\n"
                , num_lines_protected, ( float ) num_lines_protected / ( float ) m_lines.size() );

        if ( hack_use_warp_prot_list ) {
            fprintf( fout, "warps protecting each line:\n" );
            for ( std::vector< priority_cache_block * >::const_iterator it = m_lines.begin(); it != m_lines.end(); ++it ) {
                fprintf( fout, "%u, ", (*(*it)->m_warps_protecting_line.begin()).first );
            }
            fprintf( fout, "\n" );
        }
        */
    }

    //virtual enum cache_request_status probe( new_addr_type addr, unsigned &idx ) const;

    cache_request_status probe_special( new_addr_type addr, unsigned &idx, unsigned& num_protected ) const;

    unsigned get_num_protected_lines() const { return m_num_protected_lines; }
    void protect_line( unsigned cache_index, unsigned warp_id );
    bool should_whitelist_warp( unsigned warp_id ) const;

    protection_type get_protection_type() const {
        // One or the other...
        assert( ( PROTECTED_CYCLES > 0 ) ^ ( PROTECTED_ACCESES > 0 ) );
        return PROTECTED_CYCLES > 0 ? CYCLES_PROTECTION_TYPE : ACCESSES_PROTECTION_TYPE;
    }

    virtual enum cache_request_status access( new_addr_type addr, unsigned time, unsigned &idx, bool &wb, cache_block_t &evicted ) {
        for ( std::vector< priority_cache_block * >::const_iterator it = m_lines.begin(); it != m_lines.end(); ++it ) {
            if ( (*it)->m_accesses_protected > 0 ) {
                assert( get_protection_type() == ACCESSES_PROTECTION_TYPE );
                --(*it)->m_accesses_protected;
                if ( (*it)->m_accesses_protected == 0 ) {
                    --m_num_protected_lines;
                    assert( (*it)->m_warps_protecting_line.size() == 1 );
                    --m_lines_protected_per_warp[ ( ( *it )->m_warps_protecting_line.begin() )->first ].second;
                    (*it)->m_warps_protecting_line.clear();
                }
            }
        }
        assert( m_num_protected_lines <= m_lines.size() );
        cache_request_status result = base_tag_array::access( addr, time, idx, wb, evicted, NULL );

        // This is a hack to clear protection on lines that do not deserve it....
        if ( MISS == result ) {
            if ( get_protection_type() == ACCESSES_PROTECTION_TYPE ) {
                assert( m_lines[idx]->m_accesses_protected != PROTECTED_ACCESES );
            }

            if ( get_protection_type() == CYCLES_PROTECTION_TYPE
                    && m_lines[idx]->m_cycles_protected != PROTECTED_CYCLES
                    && m_lines[idx]->m_cycles_protected > 0 ) {
                m_lines[idx]->m_cycles_protected = 0;
                --m_num_protected_lines;
            } else if ( get_protection_type() == ACCESSES_PROTECTION_TYPE
                    && m_lines[idx]->m_accesses_protected != ( PROTECTED_ACCESES - 1)
                    && m_lines[idx]->m_accesses_protected > 0 ) {
                m_lines[idx]->m_accesses_protected = 0;
                --m_num_protected_lines;
            }
        }
        return result;

    }

private:
    std::vector< priority_cache_block * > m_lines;
    unsigned m_num_protected_lines;
    unsigned m_stall_override_factor;

    typedef std::pair< unsigned, unsigned > count_warp_id_pair;
    std::vector< count_warp_id_pair > m_lines_protected_per_warp;
};

class victim_tag_array : public base_tag_array {
public:
    victim_tag_array( const cache_config &config, int core_id, int type_id, bool enable_access_stream_dump )
        : base_tag_array( config,core_id,type_id, enable_access_stream_dump ) {
        m_lines.resize( config.get_num_lines(), NULL );
        for ( unsigned i = 0; i < config.get_num_lines(); ++i ) {
            base_tag_array::m_lines[ i ] = m_lines[ i ] = new victim_cache_block( config.get_line_sz() );
        }
    }

    virtual ~victim_tag_array(){
        for ( unsigned i = 0; i < m_config.get_num_lines(); ++i ) {
            delete m_lines[ i ];
        }
    }
};


class mshr_table {
public:
    mshr_table( unsigned num_entries, unsigned max_merged )
    : m_num_entries(num_entries),
    m_max_merged(max_merged),
#ifndef USE_MAP
    m_data(2*num_entries)
#endif
    {
    }

    // is there a pending request to the lower memory level already?
    bool probe( new_addr_type block_addr ) const
    {
        table::const_iterator a = m_data.find(block_addr);
        return a != m_data.end();
    }

    // is there space for tracking a new memory access?
    bool full( new_addr_type block_addr ) const 
    { 
        table::const_iterator i=m_data.find(block_addr);
        if ( i != m_data.end() )
            return i->second.size() >= m_max_merged;
        else
            return m_data.size() >= m_num_entries; 
    }

    // add or merge this access
    void add( new_addr_type block_addr, mem_fetch *mf )
    {
        m_data[block_addr].push_back(mf);
        assert( m_data.size() <= m_num_entries );
        assert( m_data[block_addr].size() <= m_max_merged );
    }

    // true if cannot accept new fill responses
    bool busy() const 
    { 
        return false;
    }

    // accept a new cache fill response: mark entry ready for processing
    void mark_ready( new_addr_type block_addr )
    {
        assert( !busy() );
        table::iterator a = m_data.find(block_addr);
        assert( a != m_data.end() ); // don't remove same request twice
        m_current_response.push_back( block_addr );
        assert( m_current_response.size() <= m_data.size() );
    }

    // true if ready accesses exist
    bool access_ready() const 
    {
        return !m_current_response.empty(); 
    }

    // next ready access
    mem_fetch *next_access()
    {
        assert( access_ready() );
        new_addr_type block_addr = m_current_response.front();
        assert( !m_data[block_addr].empty() );
        mem_fetch *result = m_data[block_addr].front();
        m_data[block_addr].pop_front();
        if ( m_data[block_addr].empty() ) {
            // release entry
            m_data.erase(block_addr); 
            m_current_response.pop_front();
        }
        return result;
    }

    void display( FILE *fp ) const
    {
        fprintf(fp,"MSHR contents\n");
        for ( table::const_iterator e=m_data.begin(); e!=m_data.end(); ++e ) {
            unsigned block_addr = e->first;
            fprintf(fp,"MSHR: tag=0x%06x, %zu entries : ", block_addr, e->second.size());
            if ( !e->second.empty() ) {
                mem_fetch *mf = e->second.front();
                fprintf(fp,"%p :",mf);
                mf->print(fp);
            } else {
                fprintf(fp," no memory requests???\n");
            }
        }
    }

private:

    // finite sized, fully associative table, with a finite maximum number of merged requests
    const unsigned m_num_entries;
    const unsigned m_max_merged;

    typedef std::list<mem_fetch*> entry;
    typedef my_hash_map<new_addr_type,entry> table;
    table m_data;

    // it may take several cycles to process the merged requests
    bool m_current_response_ready;
    std::list<new_addr_type> m_current_response;
};

class cache_t {
public:
    virtual ~cache_t() {}
    virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) =  0;
};

bool was_write_sent( const std::list<cache_event> &events );
bool was_read_sent( const std::list<cache_event> &events );

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

class tag_based_cache_t : public cache_t {
public:
    tag_based_cache_t( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport,
                     enum mem_fetch_status status, bool enable_access_dump )
    : m_config(config), m_tag_array( NULL ), m_mshrs(config.m_mshr_entries,config.m_mshr_max_merge), m_stats( config, m_tag_array ),
      m_enable_access_dump( enable_access_dump )
    {
        m_name = name;
        assert(config.m_mshr_type == ASSOC);
        m_memport=memport;
        m_miss_queue_status = status;
    }

    virtual void cycle()
    {
        // send next request to lower level of memory
        if ( !m_miss_queue.empty() ) {
            mem_fetch *mf = m_miss_queue.front();
            if ( !m_memport->full(mf->get_data_size(),mf->get_is_write(), mf) ) {
                m_miss_queue.pop_front();
                m_memport->push(mf);
            }
        }
    }

    void instant_fill_cycle( unsigned time )
    {
        for ( std::list<mem_fetch*>::iterator iter = m_miss_queue.begin();
              iter != m_miss_queue.end(); ++iter ) {
        	if ( (*iter)->get_access_type() != GLOBAL_ACC_W && (*iter)->get_access_type() != LOCAL_ACC_W ) {
        		fill( *iter, time );
        		m_mshrs.next_access();
        	}
        }
        m_miss_queue.clear();
    }

    // interface for response from lower memory level (model bandwidth restictions in caller)
    virtual void fill( mem_fetch *mf, unsigned time )
    {
        extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
        m_stats.fill_event( mf, time, e->second.m_block_addr, e->second.m_cache_index );

        assert( e != m_extra_mf_fields.end() );
        assert( e->second.m_valid );
        mf->set_data_size( e->second.m_data_size );
        if ( m_config.m_alloc_policy == ON_MISS ) {
            m_tag_array->fill(e->second.m_cache_index,time);
        }
        else if ( m_config.m_alloc_policy == ON_FILL ){
          m_tag_array->fill(e->second.m_block_addr,time);

        } else abort();
        m_mshrs.mark_ready(e->second.m_block_addr);
        m_extra_mf_fields.erase(mf);
    }

    bool waiting_for_fill( mem_fetch *mf )
    {
        extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf); 
        return e != m_extra_mf_fields.end();
    }

    // are any (accepted) accesses that had to wait for memory now ready? (does not include accesses that "HIT")
    bool access_ready() const
    {
        return m_mshrs.access_ready();
    }

    // pop next ready access (does not include accesses that "HIT")
    mem_fetch *next_access() 
    { 
        return m_mshrs.next_access(); 
    }

    // flash invalidate all entries in cache
    void flush()
    {
        m_tag_array->flush();
    }

    virtual void print(FILE *fp, unsigned &accesses, unsigned &misses) const
    {
        fprintf( fp, "Cache %s:\t", m_name.c_str() );
        m_tag_array->print(fp,accesses,misses);
        printf( "set_access_histo, " );
        for ( unsigned i = 0; i < m_config.get_num_sets(); ++i ) {
           printf( "%u, ", m_stats.get_set_access_histo( i ) );
        }
        printf( "\n" );

        printf( "set_miss_histo, " );
        for ( unsigned i = 0; i < m_config.get_num_sets(); ++i ) {
           printf( "%u, ", m_stats.get_set_miss_histo( i ) );
        }
        printf( "\n" );
        if ( m_enable_access_dump && m_tag_array->get_stats().m_access_stream.size() > 0 ) {
            char buff[ 255 ];
            buff[ sizeof( buff ) - 1 ] = '\0';
            snprintf( buff, sizeof( buff ) - 1, "%s_%s", m_name.c_str(), "access_stream.log" );
            FILE* file = fopen( buff, "w" );
            for ( std::list< access_stream_entry >::const_iterator it
                    = m_tag_array->get_stats().m_access_stream.begin();
                    it != m_tag_array->get_stats().m_access_stream.end(); ++it ) {
               it->print( file );
            }
            fclose( file );
        }
    }

    enum display_state_type {
        DISPLAY_STATE_ALL = 0,
        DISPLAY_STATE_MSHR_ONLY
    };

    void display_state( FILE *fp, display_state_type type = DISPLAY_STATE_MSHR_ONLY ) const
    {
        fprintf(fp,"Cache %s:\n", m_name.c_str() );
        m_mshrs.display(fp);
        if ( DISPLAY_STATE_ALL == type ) {
            m_tag_array->print_tag_info();
        }
        fprintf(fp,"\n");
    }

    cache_stats& get_stats() { return m_stats; }

    unsigned get_line_sz(){ return m_config.m_line_sz; }

    virtual cache_request_status probe( new_addr_type addr ) const {
        new_addr_type dummy;
        return probe( addr, dummy );
    }

    virtual cache_request_status probe( new_addr_type addr, new_addr_type& evicted_block_addr ) const {
        unsigned idx = 0xDEADBEEF;
        cache_request_status status = m_tag_array->probe( addr, idx );
        if ( status == MISS ) {
            assert( 0xDEADBEEF != idx );
            evicted_block_addr = m_tag_array->get_block( idx ).m_block_addr;
        }
        return status;
    }

protected:
    std::string m_name;
    const cache_config &m_config;
    base_tag_array*  m_tag_array;
    mshr_table m_mshrs;
    std::list<mem_fetch*> m_miss_queue;
    enum mem_fetch_status m_miss_queue_status;
    mem_fetch_interface *m_memport;

    struct extra_mf_fields {
        extra_mf_fields()  { m_valid = false;}
        extra_mf_fields( new_addr_type a, unsigned i, unsigned d ) 
        {
            m_valid = true;
            m_block_addr = a;
            m_cache_index = i;
            m_data_size = d;
        }
        bool m_valid;
        new_addr_type m_block_addr;
        unsigned m_cache_index;
        unsigned m_data_size;
    };

    typedef std::map<mem_fetch*,extra_mf_fields> extra_mf_fields_lookup;

    extra_mf_fields_lookup m_extra_mf_fields;

    cache_stats m_stats;
    bool m_enable_access_dump;
};

class read_only_cache : public tag_based_cache_t {
public:
    read_only_cache( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport, enum mem_fetch_status status,
                     bool enable_access_dump )
        : tag_based_cache_t(name,config,core_id,type_id,memport,status,enable_access_dump) {
        m_tag_array = new tag_array( config,core_id,type_id, enable_access_dump );
    }
// access cache: returns RESERVATION_FAIL if request could not be accepted (for any reason)
    virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events )
    {
        assert(m_config.m_write_policy == READ_ONLY);
        assert(!mf->get_is_write());
        new_addr_type block_addr = m_config.block_addr(addr);
        m_stats.access_event( block_addr );
        unsigned cache_index = (unsigned)-1;
        enum cache_request_status status = m_tag_array->probe(block_addr,cache_index);
        if ( status == HIT ) {
            m_tag_array->access(block_addr,time,cache_index,mf); // update LRU state
            return HIT;
        }
        if ( status != RESERVATION_FAIL ) {
            bool mshr_hit = m_mshrs.probe(block_addr);
            bool mshr_avail = !m_mshrs.full(block_addr);
            if ( mshr_hit && mshr_avail ) {
                m_tag_array->access(addr,time,cache_index,mf);
                m_mshrs.add(block_addr,mf);
                return MISS;
            } else if ( !mshr_hit && mshr_avail && (m_miss_queue.size() < m_config.m_miss_queue_size) ) {
                m_tag_array->access(addr,time,cache_index,mf);
                m_mshrs.add(block_addr,mf);
                m_extra_mf_fields[mf] = extra_mf_fields(block_addr,cache_index, mf->get_data_size());
                mf->set_data_size( m_config.get_line_sz() );
                m_miss_queue.push_back(mf);
                mf->set_status(m_miss_queue_status,time);
                events.push_back(READ_REQUEST_SENT);
                return MISS;
            }
        }
        return RESERVATION_FAIL;
    }
};

class evict_on_write_cache : public tag_based_cache_t {
public:
    virtual ~evict_on_write_cache(){}
    virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events );

protected:
    evict_on_write_cache( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport,
            mem_fetch_allocator *mfcreator, enum mem_fetch_status status, bool enable_access_dump )
            : tag_based_cache_t( name, config, core_id, type_id, memport, status, enable_access_dump ), m_memfetch_creator( mfcreator ) {
    }

    mem_fetch_allocator *m_memfetch_creator;
};

// This is meant to model the first level data cache in Fermi.
// It is write-evict (global) or write-back (local) at the granularity 
// of individual blocks (the policy used in fermi according to the CUDA manual)
class data_cache : public evict_on_write_cache {
public:
    data_cache( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport, 
                mem_fetch_allocator *mfcreator, enum mem_fetch_status status, bool enable_access_dump )
    : evict_on_write_cache(name,config,core_id,type_id,memport,mfcreator,status, enable_access_dump)
    {
        if ( config.is_rrip_policy() ) {
            m_tag_array = new rrip_tag_array( config,core_id,type_id,enable_access_dump );
        } else {
            m_tag_array = new tag_array( config,core_id,type_id, enable_access_dump );
        }
    }

    virtual ~data_cache() {
        delete m_tag_array;
    }

    void print_data_cache_info(){
    	printf("\n\n====== Data Cache Info ======\n\n");
    	printf("m_name: %s\n", m_name.c_str());
    	m_tag_array->print_tag_info();
    	printf("\n\n====== Data Cache Info ======\n\n");
    }
};

class virtual_policy_cache : public evict_on_write_cache {
public:
    virtual_policy_cache( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport,
                    mem_fetch_allocator *mfcreator, enum mem_fetch_status status, bool enable_access_dump )
        : evict_on_write_cache(name,config,core_id,type_id,memport,mfcreator,status, enable_access_dump) {
        evict_on_write_cache::m_tag_array = m_tag_array = new virtual_policy_tag_array( config, core_id, type_id, enable_access_dump );
    }

    virtual ~virtual_policy_cache() {
        delete m_tag_array;
    }

    virtual void fill( mem_fetch *mf, unsigned time ) {
        extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
        if ( mf->get_payload() ) {
            m_tag_array->get_block( e->second.m_cache_index ).m_policy = *(dynamic_cast< vm_page_mapping_payload* >( mf->get_payload() ));
        }
        evict_on_write_cache::fill( mf, time );
    }

    void invalidate_line_if_policy_does_not_match( new_addr_type addr, const vm_page_mapping_payload& configuration ) {
       unsigned cache_index = (unsigned)-1;
       new_addr_type block_addr = m_config.block_addr(addr);
       enum cache_request_status status = m_tag_array->probe( block_addr,cache_index );
       if ( ( HIT == status || HIT_RESERVED == status ) && m_tag_array->get_block(cache_index).m_policy != configuration ) {
          m_tag_array->get_block( cache_index ).m_status = INVALID;
       }
    }

    virtual enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events );

private:
    virtual_policy_tag_array* m_tag_array;
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
        m_mshrs.mark_ready(e->second.m_block_addr);
        m_extra_mf_fields.erase(mf);
    }

private:
    distill_tag_array* m_tag_array;
};

// See the following paper to understand this cache model:
// 
// Igehy, et al., Prefetching in a Texture Cache Architecture, 
// Proceedings of the 1998 Eurographics/SIGGRAPH Workshop on Graphics Hardware
// http://www-graphics.stanford.edu/papers/texture_prefetch/
class tex_cache : public cache_t {
public:
    tex_cache( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport,
               enum mem_fetch_status request_status, 
               enum mem_fetch_status rob_status )
    : m_config(config),
    m_tags(config,core_id,type_id,false),
    m_fragment_fifo(config.m_fragment_fifo_entries), 
    m_request_fifo(config.m_request_fifo_entries),
    m_rob(config.m_rob_entries),
    m_result_fifo(config.m_result_fifo_entries)
    {
        m_name = name;
        assert(config.m_mshr_type == TEX_FIFO);
        assert(config.m_write_policy == READ_ONLY);
        assert(config.m_alloc_policy == ON_MISS);
        m_memport=memport;
        m_cache = new data_block[ config.get_num_lines() ];
        m_request_queue_status = request_status;
        m_rob_status = rob_status;
    }

    // return values: RESERVATION_FAIL if request could not be accepted 
    // otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT 
    // since unlike a normal CPU cache, a "HIT" in texture cache does not 
    // mean the data is ready (still need to get through fragment fifo)
    enum cache_request_status access( new_addr_type addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) {
        if ( m_fragment_fifo.full() || m_request_fifo.full() || m_rob.full() )
            return RESERVATION_FAIL;

        // at this point, we will accept the request : access tags and immediately allocate line
        new_addr_type block_addr = m_config.block_addr(addr);
        unsigned cache_index = (unsigned)-1;
        enum cache_request_status status = m_tags.access(block_addr,time,cache_index,mf);
        assert( status != RESERVATION_FAIL );
        assert( status != HIT_RESERVED ); // as far as tags are concerned: HIT or MISS 
        m_fragment_fifo.push( fragment_entry(mf,cache_index,status==MISS,mf->get_data_size()) );
        if ( status == MISS ) {
            // we need to send a memory request...
            unsigned rob_index = m_rob.push( rob_entry(cache_index, mf, block_addr) );
            m_extra_mf_fields[mf] = extra_mf_fields(rob_index);
            mf->set_data_size(m_config.get_line_sz());
            m_tags.fill(cache_index,time); // mark block as valid 
            m_request_fifo.push(mf);
            mf->set_status(m_request_queue_status,time);
            events.push_back(READ_REQUEST_SENT);
            return MISS;
        } else {
            // the value *will* *be* in the cache already
            return HIT_RESERVED;
        }
    }

    void cycle() 
    {
        // send next request to lower level of memory
        if ( !m_request_fifo.empty() ) {
            mem_fetch *mf = m_request_fifo.peek();
            if ( !m_memport->full(mf->get_ctrl_size(),false) ) {
                m_request_fifo.pop();
                m_memport->push(mf);
            }
        }
        // read ready lines from cache
        if ( !m_fragment_fifo.empty() && !m_result_fifo.full() ) {
            const fragment_entry &e = m_fragment_fifo.peek();
            if ( e.m_miss ) {
                // check head of reorder buffer to see if data is back from memory
                unsigned rob_index = m_rob.next_pop_index();
                const rob_entry &r = m_rob.peek(rob_index);
                assert( r.m_request == e.m_request );
                assert( r.m_block_addr == m_config.block_addr(e.m_request->get_addr()) );
                if ( r.m_ready ) {
                    assert( r.m_index == e.m_cache_index );
                    m_cache[r.m_index].m_valid = true;
                    m_cache[r.m_index].m_block_addr = r.m_block_addr;
                    m_result_fifo.push(e.m_request);
                    m_rob.pop();
                    m_fragment_fifo.pop();
                }
            } else {
                // hit:
                assert( m_cache[e.m_cache_index].m_valid ); 
                assert( m_cache[e.m_cache_index].m_block_addr = m_config.block_addr(e.m_request->get_addr()) );
                m_result_fifo.push( e.m_request );
                m_fragment_fifo.pop();
            }
        }
    }

    // place returning cache block into reorder buffer
    void fill( mem_fetch *mf, unsigned time )
    {
        extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf); 
        assert( e != m_extra_mf_fields.end() );
        assert( e->second.m_valid );
        assert( !m_rob.empty() );
        mf->set_status(m_rob_status,time);

        unsigned rob_index = e->second.m_rob_index;
        rob_entry &r = m_rob.peek(rob_index);
        assert( !r.m_ready );
        r.m_ready = true;
        r.m_time = time;
        assert( r.m_block_addr == m_config.block_addr(mf->get_addr()) );
    }

    // are any (accepted) accesses that had to wait for memory now ready? (does not include accesses that "HIT")
    bool access_ready() const
    {
        return !m_result_fifo.empty();
    }

    // pop next ready access (includes both accesses that "HIT" and those that "MISS")
    mem_fetch *next_access() 
    { 
        return m_result_fifo.pop();
    }

    void display_state( FILE *fp ) const
    {
        fprintf(fp,"%s (texture cache) state:\n", m_name.c_str() );
        fprintf(fp,"fragment fifo entries  = %u / %u\n", m_fragment_fifo.size(), m_fragment_fifo.capacity() );
        fprintf(fp,"reorder buffer entries = %u / %u\n", m_rob.size(), m_rob.capacity() );
        fprintf(fp,"request fifo entries   = %u / %u\n", m_request_fifo.size(), m_request_fifo.capacity() );
        if ( !m_rob.empty() )
            fprintf(fp,"reorder buffer contents:\n");
        for ( int n=m_rob.size()-1; n>=0; n-- ) {
            unsigned index = (m_rob.next_pop_index() + n)%m_rob.capacity();
            const rob_entry &r = m_rob.peek(index);
            fprintf(fp, "tex rob[%3d] : %s ", index, (r.m_ready?"ready  ":"pending") );
            if ( r.m_ready )
                fprintf(fp,"@%6u", r.m_time );
            else
                fprintf(fp,"       ");
            fprintf(fp,"[idx=%4u]",r.m_index);
            r.m_request->print(fp,false);
        }
        if ( !m_fragment_fifo.empty() ) {
            fprintf(fp,"fragment fifo (oldest) :");
            fragment_entry &f = m_fragment_fifo.peek();
            fprintf(fp,"%s:          ", f.m_miss?"miss":"hit ");
            f.m_request->print(fp,false);
        }
    }


    private:
    std::string m_name;
    const cache_config &m_config;

    struct fragment_entry {
        fragment_entry() {}
        fragment_entry( mem_fetch *mf, unsigned idx, bool m, unsigned d )
        {
            m_request=mf;
            m_cache_index=idx;
            m_miss=m;
            m_data_size=d;
        }
        mem_fetch *m_request;     // request information
        unsigned   m_cache_index; // where to look for data
        bool       m_miss;        // true if sent memory request
        unsigned   m_data_size;
    };

    struct rob_entry {
        rob_entry() { m_ready = false; m_time=0; m_request=NULL;}
        rob_entry( unsigned i, mem_fetch *mf, new_addr_type a ) 
        { 
            m_ready=false; 
            m_index=i;
            m_time=0;
            m_request=mf; 
            m_block_addr=a;
        }
        bool m_ready;
        unsigned m_time; // which cycle did this entry become ready?
        unsigned m_index; // where in cache should block be placed?
        mem_fetch *m_request;
        new_addr_type m_block_addr;
    };

    struct data_block {
        data_block() { m_valid = false;}
        bool m_valid;
        new_addr_type m_block_addr;
    };

    // TODO: replace fifo_pipeline with this?
    template<class T> class fifo {
    public:
        fifo( unsigned size ) 
        { 
            m_size=size; 
            m_num=0; 
            m_head=0; 
            m_tail=0; 
            m_data = new T[size];
        }
        bool full() const { return m_num == m_size;}
        bool empty() const { return m_num == 0;}
        unsigned size() const { return m_num;}
        unsigned capacity() const { return m_size;}
        unsigned push( const T &e ) 
        { 
            assert(!full()); 
            m_data[m_head] = e; 
            unsigned result = m_head;
            inc_head(); 
            return result;
        }
        T pop() 
        { 
            assert(!empty()); 
            T result = m_data[m_tail];
            inc_tail();
            return result;
        }
        const T &peek( unsigned index ) const 
        { 
            assert( index < m_size );
            return m_data[index]; 
        }
        T &peek( unsigned index ) 
        { 
            assert( index < m_size );
            return m_data[index]; 
        }
        T &peek() const
        { 
            return m_data[m_tail]; 
        }
        unsigned next_pop_index() const 
        {
            return m_tail;
        }
    private:
        void inc_head() { m_head = (m_head+1)%m_size; m_num++;}
        void inc_tail() { assert(m_num>0); m_tail = (m_tail+1)%m_size; m_num--;}

        unsigned   m_head; // next entry goes here
        unsigned   m_tail; // oldest entry found here
        unsigned   m_num;  // how many in fifo?
        unsigned   m_size; // maximum number of entries in fifo
        T         *m_data;
    };

    tag_array               m_tags;
    fifo<fragment_entry>    m_fragment_fifo;
    fifo<mem_fetch*>        m_request_fifo;
    fifo<rob_entry>         m_rob;
    data_block             *m_cache;
    fifo<mem_fetch*>        m_result_fifo; // next completed texture fetch

    mem_fetch_interface    *m_memport;
    enum mem_fetch_status   m_request_queue_status;
    enum mem_fetch_status   m_rob_status;

    struct extra_mf_fields {
        extra_mf_fields()  { m_valid = false;}
        extra_mf_fields( unsigned i ) 
        {
            m_valid = true;
            m_rob_index = i;
        }
        bool m_valid;
        unsigned m_rob_index;
    };

    typedef std::map<mem_fetch*,extra_mf_fields> extra_mf_fields_lookup;

    extra_mf_fields_lookup m_extra_mf_fields;
};

#define MAX_REGION_SIZE 2048

class vm_manager_config : public cache_config {
public:
    enum obj_size_policy {
       OBJ_SIZE_POLICY_TYPE_INVALID = 0,
       OBJ_SIZE_POLICY_TYPE_GCD,
       OBJ_SIZE_POLICY_TYPE_BYTE_LINE_INTERPRETATION,
       OBJ_SIZE_POLICY_TYPE_MAX
    };

    enum update_policy {
        UPDATE_POLICY_USE_REGION_CACHE,
        UPDATE_POLICY_DIRECT_TO_TLB,
        UPDATE_POLICY_MAX
    };

    vm_manager_config() : m_object_size_policy( OBJ_SIZE_POLICY_TYPE_INVALID ) { }

    virtual void init()
    {
        unsigned enabled=0;
        if ( strncmp( m_config_string, "none", 4 ) == 0 ) {
            m_valid = false;
            m_enabled = false;
            return;
        }
        sscanf(m_config_string,"%u:%u:%u:%u", &enabled, &m_threshold, &m_word_size, &m_max_tile_size);
        m_enabled=(enabled!=0);
        m_object_size_policy = m_enabled ? obj_size_policy( enabled ) : OBJ_SIZE_POLICY_TYPE_INVALID;
        assert( m_object_size_policy >= OBJ_SIZE_POLICY_TYPE_INVALID && m_object_size_policy < OBJ_SIZE_POLICY_TYPE_MAX  );
        const char *config = strchr(m_config_string,',');
        assert(config);
        cache_config::init(++config);
        if( region_size() > MAX_REGION_SIZE ) {
            printf("GPGPU-Sim ERROR ** vm_manager_config region size %u larger than MAX_REGION_SIZE (%u)\n", region_size(), MAX_REGION_SIZE );
            abort();
        }

        m_update_policy = UPDATE_POLICY_DIRECT_TO_TLB; //TODO allow this to be runtime configurable
    }

    unsigned region_size() const { return get_line_sz() / m_word_size; }

    unsigned m_threshold;
    unsigned m_word_size;
    unsigned m_max_tile_size; 

    bool m_enabled;

    obj_size_policy m_object_size_policy;
    update_policy m_update_policy;
};

class vm_policy_manager_stats {
public:
    vm_policy_manager_stats()
        : m_num_accesses_with_no_object_size( 0 ),
          m_num_attempted_insertions( 0 ),
          m_num_insertions_that_already_exist( 0 ) {}

    void print( FILE* fp ) const {
        fprintf( fp, "vm_policy_manager_stats::m_num_accesses_with_no_object_size=%u\n",m_num_accesses_with_no_object_size );
        fprintf( fp, "vm_policy_manager_stats::m_num_attempted_insertions=%u\n",m_num_attempted_insertions );
        fprintf( fp, "vm_policy_manager_stats::m_num_insertions_that_already_exist=%u\n",m_num_insertions_that_already_exist );
    }

    void attemtped_policy_event( const vm_page_mapping_payload* best, size_t object_size ) {
        if ( best ) {
            ++m_num_attempted_insertions;
        }

        if ( object_size == 0 ) {
            ++m_num_accesses_with_no_object_size;
        }
    }

    void redundant_policy_event() {
        ++m_num_insertions_that_already_exist;
    }

private:
    unsigned m_num_accesses_with_no_object_size;
    unsigned m_num_attempted_insertions;
    unsigned m_num_insertions_that_already_exist;
};

// All the dynamic determination of page properties is done here
// Jobs:
//  1) Dynamically Determine Object Size through two possible methods:
//      (a) GCD within the addresses in the instruction (POLICY_TYPE_GCD)
//      (b) From a bit vector of accesses to a "region" i.e. a page (POLICY_TYPE_BYTE_LINE_INTERPRETATION) The bit vector is stored in m_data and is sized
//              to the cache dimensions
// Based on which page format we are using it determines the following ( Only one of the following can be true )
//  if VP_TILED - Note, this assumes completely aligned data
//      2) The field size used (currently just passed as a config option, although the byte access line might be used to determine this)
//      3) The tile width to use (just now just tiles the whole page)
//  if VP_HOT_DATA_ALIGNED - Note, this assumes completely aligned data
//      2) The hot bytes in each object - m_obj_size_to_hit_vector stores this for all our different object sizes.
//  if VP_HOT_DATA_UNALIGNED
//      2) The hot bytes in each object - m_obj_size_to_hit_vector stores this for all our different object sizes.
//      3) The start of our "first page", used as the compass for all other accesses to this array
class vm_policy_manager {
public:
    vm_policy_manager( const vm_manager_config &config, 
                       mem_fetch_allocator *mf_allocator, 
                       mem_fetch_interface *icnt, 
                       enum mem_fetch_status status,
                       unsigned core_id,
                       size_t l1_line_size,
                       virtual_page_factory* factory )
        :   m_page_factory( factory ),
            m_config(config),
            m_tags(config,-1,-1,false),
            m_l1_cache_line_size( l1_line_size )

    {
        m_data = new region_entry[ config.get_num_lines() ];
        m_icnt = icnt;
        m_send_status = status;
        m_mf_allocator=mf_allocator;
        m_core_id = core_id;
    }
    void access( warp_inst_t& inst, unsigned time, shader_page_reordering_tlb& tlb );
    vm_page_mapping_payload *cycle();
    void fill( mem_fetch *mf );
    void update_object_access_vector( const warp_inst_t& inst );

    void print( FILE* fp ) const;

private:
    static const size_t INVALID_OBJECT_SIZE = 0xDEADBEEFDEADBEEF;

    struct region_entry {
        region_entry() { m_count=0; m_pending_change=false; }

        unsigned access(unsigned offset)
        {
            m_accessed.set(offset);
            m_count++;
            return m_count;
        }
       
        unsigned m_count;
        std::bitset<MAX_REGION_SIZE> m_accessed;
        bool m_pending_change;
    };

    void print_object_access_info( FILE* fp ) const;

    size_t determine_gcd_obj_size( const warp_inst_t& instr ) const;
    size_t gcd( size_t a, size_t b ) const ;

    virtual_page_factory* m_page_factory;

    // for transpose vm stuff
    const vm_manager_config &m_config;
    tag_array m_tags;
    region_entry *m_data;

    vm_page_mapping_payload select_tiled_mapping( new_addr_type block_addr, size_t obj_size );

    unsigned find_hot_data( boost::dynamic_bitset<>& hotmap, size_t object_size ) const;
    vm_page_mapping_payload select_aligned_hot_mapping( new_addr_type block_addr, size_t obj_size );
    vm_page_mapping_payload select_unaligned_hot_mapping( new_addr_type block_addr, size_t obj_size );

    size_t determine_region_access_vector_obj_size( const struct region_entry &region ) const;
    void update_region_access_vector( warp_inst_t& inst, unsigned time );

    vm_page_mapping_payload get_no_translation( new_addr_type block_addr, size_t obj_size ) const;
    void handle_new_policy( vm_page_mapping_payload* best, new_addr_type block_addr, region_entry &e, shader_page_reordering_tlb& tlb );

    // communication to L2 and TLB
    mem_fetch_interface *m_icnt;
    enum mem_fetch_status m_send_status;
    std::list<mem_fetch*> m_request_fifo;
    mem_fetch_allocator *m_mf_allocator;
    unsigned m_core_id;
    std::list<vm_page_mapping_payload*> m_tlb_update_queue;

    // hot element profiling stuff
    size_t m_l1_cache_line_size;
    struct hit_vec_and_start_of_first_page {
       std::vector< unsigned > vec;
       new_addr_type start_of_first_page;
    };
    std::map< size_t, hit_vec_and_start_of_first_page > m_obj_size_to_hit_vector;
    vm_policy_manager_stats m_stats;
};

class region_cache_config : public cache_config {
public:
    region_cache_config() { }

    virtual void init()
    {
        unsigned enabled=0;
        if ( strncmp( m_config_string, "none", 4 ) == 0 ) {
            m_valid = false;
            m_enabled = false;
            return;
        }
        sscanf(m_config_string,"%u", &enabled);
        const char *config = strchr(m_config_string,',');
        assert(config);
        cache_config::init(++config);
        m_enabled = (enabled != 0);
    }
    bool enabled() const { return m_enabled; }
private:
    bool m_enabled;
};

class region_cache {
public:
    region_cache( const region_cache_config &config ) : m_config(config), m_tags(config,-1,-1,false)
    {
        m_data = new vm_page_mapping_payload[ config.get_num_lines() ];
    }

    enum cache_request_status access( mem_fetch *mf, unsigned time, vm_page_mapping_payload &region_policy )
    {
        unsigned idx;
        enum cache_request_status status = m_tags.probe(mf->get_addr(),idx);
        if( status == HIT ) {
            region_policy = m_data[idx];
            m_tags.access(mf->get_addr(),time,idx,mf); // update LRU stack
        }
        return status;
    }
    void fill( mem_fetch *mf, unsigned time, vm_page_mapping_payload &region_policy, bool &evicted, vm_page_mapping_payload &evicted_policy )
    {
       unsigned idx;
       enum cache_request_status status = m_tags.probe(mf->get_addr(),idx);
       if( status == MISS ) {
          if( m_tags.get_block(idx).m_status == VALID ) {
              evicted = true;
              evicted_policy = m_data[idx];
          }
          m_tags.fill( mf->get_addr(), time );
       }
       m_data[idx] = region_policy;
    }
    void invalidate( mem_fetch *mf )
    {
        unsigned idx;
        enum cache_request_status status = m_tags.probe(mf->get_addr(),idx);
        assert( status == HIT );
        m_tags.get_block(idx).m_status = INVALID;
    }
private:
    const region_cache_config &m_config;
    tag_array m_tags;
    vm_page_mapping_payload *m_data;
    std::list<mem_fetch*> m_response_queue;
};

#if 0
class dynamic_tile_data_cache : public data_cache {
public:
    dynamic_tile_data_cache( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport, 
                mem_fetch_allocator *mfcreator, enum mem_fetch_status status ) 
    : data_cache(name,config,core_id,type_id,memport,mfcreator,status)
    {
    }

    virtual enum cache_request_status access( new_addr_type m_addr, mem_fetch *mf, unsigned time, std::list<cache_event> &events ) 
    {
        mem_fetch_payload *p = mf->get_payload();
        vm_page_mapping *req_mapping = dynamic_cast<vm_page_mapping*>(p);

        if( req_mapping->get_translation_config() == DYNAMIC_TILED_TRANSLATION ) {
            new_addr_type block_addr = m_config.block_addr(m_addr);
            unsigned cache_index = (unsigned)-1;
            enum cache_request_status status = m_tag_array->probe(block_addr,cache_index);
            if( status == HIT ) {
                cache_block_t &blk = m_tag_array->get_block(cache_index);
                if( blk.m_policy != *req_mapping ) 
                    status = MISS;
                else {
                    // check if specific requested stuff is in block...
                    
                    
                }
            }
            if( status == MISS ) {
                // do miss processing...
                // send request to L2...
            } else {
                // do hit processing...
            }
        }
    }
};
#endif

#endif
