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

#ifndef MC_PARTITION_INCLUDED
#define MC_PARTITION_INCLUDED

#include "dram.h"
#include "../abstract_hardware_model.h"
#include "gpu-misc.h"

#include <list>
#include <queue>

class mem_fetch;

typedef struct region_map{
	region_map() : ref_count(0), num_lifetimes(0), avg_useful(0.0), tot_ref_count(0) {}
	unsigned ref_count;
	std::vector<bool> used_blocks;
	
	unsigned num_lifetimes;
	float avg_useful;
	unsigned tot_ref_count;
}region_map;

class region_metric {
// Keeps track of different regions passing between L2 and DRAM
public:
	region_metric(unsigned ln_sz, unsigned reg_sz){
		region_sz = reg_sz;
		log2_region_sz = LOGB2(reg_sz);
		line_sz = ln_sz;
		log2_line_sz = LOGB2(ln_sz);			
		avg_fraction_useful = 0;
		num_lifetimes = 0;
		num_blocks_per_region = (reg_sz / ln_sz); // Assume power of 2 initially
		temp_sim_cycle = 0;

		avg_fraction_useful_bin = new unsigned[num_blocks_per_region];
		for(unsigned i=0; i<num_blocks_per_region; i++)
			avg_fraction_useful_bin[i] = 0;
	//printf("\n\nreg_sz: %u, log_reg_sz: %u, line_sz: %u, log_line_sz: %u, num_blks/reg: %u\n\n", region_sz, log2_region_sz, line_sz, log2_line_sz, num_blocks_per_region);
	}
	
	void add_block(new_addr_type addr){
		new_addr_type reg = get_reg_addr(addr);
		unsigned block = get_block_num(addr);
		assert(block < num_blocks_per_region);
		std::map<new_addr_type, region_map>::iterator i = reg_map.find(reg);
		region_map *entry = NULL;
	
		if(i == reg_map.end()){ // Not in region map
			entry = &reg_map[reg];
			for(unsigned j=0; j<num_blocks_per_region; j++)
				entry->used_blocks.push_back(false);
				
		}else{ // Region is already in map
			entry = &i->second;
		}
		entry->ref_count++; 
		entry->tot_ref_count++;
		entry->used_blocks.at(block) = true; // Set block as used - May already be set

	}

	void remove_block(new_addr_type addr){
		new_addr_type reg = get_reg_addr(addr);
		std::map<new_addr_type, region_map>::iterator i = reg_map.find(reg);
		region_map *entry = NULL;
		
		assert(i != reg_map.end()); // If you're removing it, it should already be in the map
		entry = &i->second;
		entry->ref_count--;

		if(entry->ref_count == 0){ // Last block of region in region map
			num_lifetimes++;	// Region will be evicted -> lifetime over
			unsigned used_count = 0;
			for(unsigned j=0; j<num_blocks_per_region; j++){
				if(entry->used_blocks.at(j) == true){
					used_count++;
					entry->used_blocks.at(j) = false;
				}
			}
			assert(used_count <= num_blocks_per_region);
			assert(used_count > 0);
			entry->num_lifetimes++;
			entry->avg_useful += ( (float)used_count/(float)num_blocks_per_region );
			avg_fraction_useful += ( (float)used_count/(float)num_blocks_per_region );
			avg_fraction_useful_bin[used_count-1] += used_count;
			//reg_map.erase(reg);
		}
	}
/*
	void flush_region_metric(){
		std::map<new_addr_type, region_map>::iterator i = reg_map.begin();
		region_map *entry = NULL;
		for(;i != reg_map.end(); i++){
			entry = &i->second;
			unsigned used_count = 0;
			for(unsigned j=0; j<num_blocks_per_region; j++){
				if(entry->used_blocks.at(j) == true)
					used_count++;
			}
			num_lifetimes++;
			avg_fraction_useful += ( (float)used_count/(float)num_blocks_per_region );
		}
	}
*/
	void set_sim_cycle(unsigned long long t){
		temp_sim_cycle = t;
	}

	unsigned long long get_sim_cycle(){
		return temp_sim_cycle;
	}

	void print_region_metric_info( unsigned id){
		// Add up what is currently in the region_metric_cache
		unsigned lifetimes = 0;
		float avg_fraction = 0.0;
		unsigned *bin = new unsigned[num_blocks_per_region];

		for(unsigned i=0; i<num_blocks_per_region; i++)
			bin[i] = 0;

		std::map<new_addr_type, region_map>::iterator i = reg_map.begin();
		region_map *entry = NULL;
		for(;i != reg_map.end(); i++){
			entry = &i->second;
			unsigned used_count = 0;
			for(unsigned j=0; j<num_blocks_per_region; j++){
				if(entry->used_blocks.at(j) == true)
					used_count++;
			}
			lifetimes++;
			avg_fraction += ( (float)used_count/(float)num_blocks_per_region );
			bin[used_count-1] += used_count;
		}
		lifetimes += num_lifetimes;
		avg_fraction += avg_fraction_useful;
		unsigned total = 0;
		for(unsigned j=0; j<num_blocks_per_region; j++){
			bin[j] += avg_fraction_useful_bin[j];
			total += bin[j];
		}

		printf("mem_part_unit: %u	Total number of lifetimes for L2 cache regions: %u\n", id, lifetimes);
		printf("mem_part_unit: %u	Average fraction of L2 cache region usefulness: %.2lf\n	",id,  avg_fraction);
		for(unsigned j=0; j<num_blocks_per_region; j++){
			printf("%.2lf	", (float)bin[j]/(float)total);
		}
		printf("\n\n");
	}

	void print_specific_region_info(unsigned id){
		std::map<new_addr_type, region_map>::iterator i = reg_map.begin();
		region_map *entry = NULL;
		unsigned num_regions=0;
		unsigned tot_num_lifetimes = 0;
		float tot_fraction_useful = 0.0;
		unsigned tot_references = 0;
		
		unsigned max_ref = 0;
		unsigned min_ref = 0xFFFFFFFF;
		for(;i != reg_map.end(); i++){
			num_regions++;
			entry = &i->second;
			unsigned used_count = 0;
			for(unsigned j=0; j<num_blocks_per_region; j++){
				if(entry->used_blocks.at(j) == true)
					used_count++;
			}
			tot_fraction_useful += (entry->avg_useful + (float)used_count/(float)num_blocks_per_region);
			tot_num_lifetimes += (entry->num_lifetimes + 1);	
			tot_references += entry->tot_ref_count;
		
			if(entry->tot_ref_count > max_ref)
				max_ref = entry->tot_ref_count;
			if(entry->tot_ref_count < min_ref)
				min_ref = entry->tot_ref_count;
		}
		
		float avg_num_lifetimes = (float)tot_num_lifetimes / (float)num_regions;
		float avg_fract_useful = tot_fraction_useful / (float)num_regions;
		float avg_ref = (float)tot_references / (float)num_regions;
		if(id == 0)
			printf("# regions: %u\n", num_regions);
	/*
		printf("mem_part_unit: %u	Max_ref: %u	Min_ref: %u\n",id, max_ref, min_ref);
		printf("mem_part_unit: %u	Average number of lifetimes for L2 cache regions:	%.2lf\n", id, avg_num_lifetimes);
		printf("mem_part_unit: %u	Average fraction of L2 cache region usefulness:	%.2lf\n",id,  avg_fract_useful);
		printf("mem_part_unit: %u	Average number of references for L2 cache regions:	%.2lf\n", id, avg_ref);
	*/
		printf("%u,	%.2lf,	%.2lf,	%.2lf,	%u,	%u\n", id, avg_num_lifetimes, avg_fract_useful, avg_ref, max_ref, min_ref);
	}
private:
	unsigned get_block_num(new_addr_type addr){
		// Gets the block number within a region
		return (addr >> log2_line_sz) & (num_blocks_per_region - 1);
	}

	new_addr_type get_reg_addr(new_addr_type addr){
		return addr >> log2_region_sz;
	}

	float avg_fraction_useful;
	unsigned num_lifetimes;
	unsigned region_sz;
	unsigned line_sz;
	unsigned log2_line_sz;
	unsigned log2_region_sz;
	unsigned num_blocks_per_region;
	std::map<new_addr_type, region_map> reg_map;
	unsigned long long temp_sim_cycle;
	
	unsigned *avg_fraction_useful_bin; 

};


class partition_mf_allocator : public mem_fetch_allocator {
public:
    partition_mf_allocator( const memory_config *config )
    {
        m_memory_config = config;
    }
    virtual mem_fetch * alloc(const class warp_inst_t &inst, const mem_access_t &access, mem_fetch_payload *payload) const
    {
        abort();
        return NULL;
    }
    virtual mem_fetch * alloc(new_addr_type addr, mem_access_type type, unsigned size, bool wr, mem_fetch_payload *payload) const;
    mem_fetch *alloc( unsigned sid, unsigned cluster_id, new_addr_type addr, mem_access_type type, mem_fetch_payload *payload ) const;
    mem_fetch *alloc(unsigned sid, unsigned cluster_id, new_addr_type addr, mem_access_type type, unsigned size, bool wr, mem_fetch_payload *payload ) const;
private:
    const memory_config *m_memory_config;
};

class memory_partition_unit 
{
public:
   memory_partition_unit( unsigned partition_id, const struct memory_config *config, class memory_stats_t *stats );
   ~memory_partition_unit(); 

   bool busy() const;

   void cache_cycle( unsigned cycle );
   void dram_cycle();

   bool full() const;
   void push( class mem_fetch* mf, unsigned long long clock_cycle );
   class mem_fetch* pop(); 
   class mem_fetch* top();
   void set_done( mem_fetch *mf );

   unsigned flushL2();

   void visualizer_print( gzFile visualizer_file );
   void print_cache_stat(unsigned &accesses, unsigned &misses) const;
   void print_stat( FILE *fp ) { m_dram->print_stat(fp); }
   void visualize() const { m_dram->visualize(); }
   void print( FILE *fp ) const;

   //void flush_region_metric(){m_region_metric->flush_region_metric();}
   void print_region_metric(unsigned id) {m_region_metric->print_region_metric_info(id);}
   void print_specific_region_metric(unsigned id) {m_region_metric->print_specific_region_info(id);}

private:
    void region_cache_policy_update( mem_fetch *mf );
    void region_cache_policy_check( mem_fetch *mf );

// data
   unsigned m_id;
   const struct memory_config *m_config;
   class dram_t *m_dram;
   
   
   class evict_on_write_cache *m_L2cache;

   class L2interface *m_L2interface;
   partition_mf_allocator *m_mf_allocator;
   class region_cache *m_vm_region_cache;

   // model delay of ROP units with a fixed latency
   struct rop_delay_t
   {
    	unsigned long long ready_cycle;
    	class mem_fetch* req;
   };
   std::queue<rop_delay_t> m_rop;

   // model DRAM access scheduler latency (fixed latency between L2 and DRAM)
   struct dram_delay_t
   {
      unsigned long long ready_cycle;
      class mem_fetch* req;
   };
   std::queue<dram_delay_t> m_dram_latency_queue;

   // these are various FIFOs between units within a memory partition
   fifo_pipeline<mem_fetch> *m_icnt_L2_queue;
   fifo_pipeline<mem_fetch> *m_L2_dram_queue;
   fifo_pipeline<mem_fetch> *m_dram_L2_queue;
   fifo_pipeline<mem_fetch> *m_L2_icnt_queue; // L2 cache hit response queue

   class mem_fetch *L2dramout; 
   unsigned long long int wb_addr;

   class memory_stats_t *m_stats;

   std::set<mem_fetch*> m_request_tracker;

   friend class L2interface;


   class region_metric *m_region_metric;
};

class L2interface : public mem_fetch_interface {
public:
    L2interface( memory_partition_unit *unit ) { m_unit=unit; }
    virtual bool full( unsigned size, bool write, mem_fetch* mf = NULL ) const
    {
       if ( mf && dynamic_cast< vm_page_mapping_payload* >( mf->get_payload() )
             && dynamic_cast< vm_page_mapping_payload* >( mf->get_payload() )->get_page_type() != VP_NON_TRANSLATED ) {
          size_t num_fetches = reverse_transform_mf_to_mf_list( mf ).size();
          return m_unit->m_L2_dram_queue->get_n_element() + num_fetches >= m_unit->m_L2_dram_queue->get_max_len();
       } else {
         // assume read and write packets all same size
         return m_unit->m_L2_dram_queue->full();
       }
    }
    virtual void push(mem_fetch *mf);
private:
   std::list< mem_fetch* > reverse_transform_mf_to_mf_list( mem_fetch* base_memfetch ) const;
   memory_partition_unit *m_unit;
};



#endif
