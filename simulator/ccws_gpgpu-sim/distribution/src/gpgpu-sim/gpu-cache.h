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
#include "../abstract_hardware_model.h"
#include "../tr1_hash_map.h"

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
    WRITE_REQUEST_SENT,
    VC_HIT
};

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

enum replacement_policy_t {
    LRU,
    FIFO,
    DIVERGENT_MRU,
    STATIC_LOAD_PREDICTION,
    START_RRIP,
    SRRIP = START_RRIP,
    BRRIP,
    END_RRIP = BRRIP
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
       	return addr >> LOGB2(WOC_WORD_SIZE) & ((WOC_WORD_SIZE/2)-1);
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
        char rw, type, status;
        m_mem_space_type = NUM_MEM_ACCESS_TYPE;
        const unsigned ntok = sscanf( str, "%llx:%c:%c:%c:%u:%zu:%llu", &m_addr, &rw, &type, &status,
                                      &m_warp_id, &m_num_threads_touching, &m_pc );
        if ( ntok != 4 ) {
            char flush;
            assert( sscanf( str, "%c", &flush ) == 1 );
            if ( 'F' == flush ) {
                m_addr = CACHE_FLUSH_SIGNATURE;
            } else {
                fprintf( stderr, "Error - Unknown Entry\n" );
                abort();
            }
        } else {
            assert( ntok == 4 );
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

    void print( FILE* output ) const {
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
           fprintf( output, "%llx:%c:%c:%c:%u:%zu:%llu\n", m_addr, rw, entry, cache_status,
                    m_warp_id, m_num_threads_touching, m_pc );
        } else {
          fprintf( output, "F\n" );
        }
    }

    new_addr_type get_addr() const { return m_addr; }
    mem_access_type get_mem_space_type() const { return m_mem_space_type; }
    bool is_cache_flush() const { return CACHE_FLUSH_SIGNATURE == m_addr; }

    static const new_addr_type CACHE_FLUSH_SIGNATURE = 0xDEADBEEFFEEBDAED;
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
    float windowed_miss_rate( ) const;
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
            return i->second.m_list.size() >= m_max_merged;
        else
            return m_data.size() >= m_num_entries; 
    }

    // add or merge this access
    void add( new_addr_type block_addr, mem_fetch *mf )
    {
        m_data[block_addr].m_list.push_back(mf);
        assert( m_data.size() <= m_num_entries );
        assert( m_data[block_addr].m_list.size() <= m_max_merged );
        // indicate that this MSHR entry contains an atomic operation 
        if ( mf->isatomic() ) {
            m_data[block_addr].m_has_atomic = true; 
        }
    }

    // true if cannot accept new fill responses
    bool busy() const 
    { 
        return false;
    }

    // accept a new cache fill response: mark entry ready for processing
    void mark_ready( new_addr_type block_addr, bool &has_atomic )
    {
        assert( !busy() );
        table::iterator a = m_data.find(block_addr);
        assert( a != m_data.end() ); // don't remove same request twice
        m_current_response.push_back( block_addr );
        has_atomic = a->second.m_has_atomic; 
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
        assert( !m_data[block_addr].m_list.empty() );
        mem_fetch *result = m_data[block_addr].m_list.front();
        m_data[block_addr].m_list.pop_front();
        if ( m_data[block_addr].m_list.empty() ) {
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
            fprintf(fp,"MSHR: tag=0x%06x, atomic=%d %zu entries : ", block_addr, e->second.m_has_atomic, e->second.m_list.size());
            if ( !e->second.m_list.empty() ) {
                mem_fetch *mf = e->second.m_list.front();
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

    struct mshr_entry {
        std::list<mem_fetch*> m_list;
        bool m_has_atomic; 
        mshr_entry() : m_has_atomic(false) { }
    }; 
    typedef my_hash_map<new_addr_type,mshr_entry> table;
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

bool was_vc_hit( const std::list<cache_event> &events );
bool was_write_sent( const std::list<cache_event> &events );
bool was_read_sent( const std::list<cache_event> &events );

class tag_based_cache_t : public cache_t {
public:
    tag_based_cache_t( const char *name, const cache_config &config, int core_id, int type_id, mem_fetch_interface *memport,
                     enum mem_fetch_status status, bool enable_access_dump )
    : m_config(config), m_tag_array( NULL ), m_mshrs(config.m_mshr_entries,config.m_mshr_max_merge), m_stats( config, m_tag_array ),
      m_enable_access_dump( enable_access_dump ), m_has_dumped_access_stream(false), m_core_id(core_id)
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
        dump_access_data( 2000 );
    }

    void dump_access_data( unsigned minimum_entries )
    {
        if ( m_enable_access_dump && m_tag_array->get_stats().m_access_stream.size() > minimum_entries ) {
            char buff[ 255 ];
            buff[ sizeof( buff ) - 1 ] = '\0';
            snprintf( buff, sizeof( buff ) - 1, "%s_%s", m_name.c_str(), "access_stream.log" );
            FILE* file = NULL;
            if ( !m_has_dumped_access_stream ) {
                file = fopen( buff, "w" );
                m_has_dumped_access_stream = true;
            } else {
                file = fopen( buff, "a" );
            }
            for ( std::list< access_stream_entry >::const_iterator it
                    = m_tag_array->get_stats().m_access_stream.begin();
                    it != m_tag_array->get_stats().m_access_stream.end(); ++it ) {
               it->print( file );
            }
            fclose( file );
            m_tag_array->get_stats().m_access_stream.clear();
        }
    }

    // interface for response from lower memory level (model bandwidth restictions in caller)
    virtual void fill( mem_fetch *mf, unsigned time )
    {
        extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
        m_stats.fill_event( mf, time, e->second.m_block_addr, e->second.m_cache_index );

        assert( e != m_extra_mf_fields.end() );
        assert( e->second.m_valid );
        mf->set_data_size( e->second.m_data_size );
        if ( m_config.m_alloc_policy == ON_MISS )
            m_tag_array->fill(e->second.m_cache_index,time);
        else if ( m_config.m_alloc_policy == ON_FILL )
            m_tag_array->fill(e->second.m_block_addr,time);
        else abort();
        bool has_atomic = false; 
        m_mshrs.mark_ready(e->second.m_block_addr, has_atomic);
        if (has_atomic) {
            assert(m_config.m_alloc_policy == ON_MISS); 
            cache_block_t &block = m_tag_array->get_block(e->second.m_cache_index); 
            block.m_status = MODIFIED; // mark line as dirty for atomic operation 
        }
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
    virtual void flush()
    {
        m_tag_array->flush();
    }

    virtual void print(FILE *fp, unsigned &accesses, unsigned &misses)
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
        dump_access_data( 0 );
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
    bool m_has_dumped_access_stream;
    int m_core_id;
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
        assert( mf->get_data_size() <= m_config.get_line_sz());

        enum mem_access_type type = mf->get_access_type();
        assert( CONST_ACC_R == type || INST_ACC_R == type );

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

        assert( mf->get_data_size() <= m_config.get_line_sz());

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


#endif
