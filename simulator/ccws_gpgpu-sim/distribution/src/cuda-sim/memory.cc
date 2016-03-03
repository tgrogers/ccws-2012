// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung,
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

#include "memory.h"
#include <stdlib.h>
#include "../debug.h"

template<unsigned BSIZE> memory_space_impl<BSIZE>::memory_space_impl( std::string name, unsigned hash_size )
{
   m_name = name;
   MEM_MAP_RESIZE(hash_size);

   m_log2_block_size = -1;
   for( unsigned n=0, mask=1; mask != 0; mask <<= 1, n++ ) {
      if( BSIZE & mask ) {
         assert( m_log2_block_size == (unsigned)-1 );
         m_log2_block_size = n; 
      }
   }
   assert( m_log2_block_size != (unsigned)-1 );
}

template<unsigned BSIZE> void memory_space_impl<BSIZE>::write( mem_addr_t addr, size_t length, const void *data, class ptx_thread_info *thd, const ptx_instruction *pI)
{
   mem_addr_t index = addr >> m_log2_block_size;
   if ( (addr+length) <= (index+1)*BSIZE ) {
      // fast route for intra-block access 
      unsigned offset = addr & (BSIZE-1);
      unsigned nbytes = length;
      m_data[index].write(offset,nbytes,(const unsigned char*)data);
   } else {
      // slow route for inter-block access
      unsigned nbytes_remain = length;
      unsigned src_offset = 0; 
      mem_addr_t current_addr = addr; 

      while (nbytes_remain > 0) {
         unsigned offset = current_addr & (BSIZE-1);
         mem_addr_t page = current_addr >> m_log2_block_size; 
         mem_addr_t access_limit = offset + nbytes_remain; 
         if (access_limit > BSIZE) {
            access_limit = BSIZE;
         } 
         
         size_t tx_bytes = access_limit - offset; 
         m_data[page].write(offset, tx_bytes, &((const unsigned char*)data)[src_offset]);

         // advance pointers 
         src_offset += tx_bytes; 
         current_addr += tx_bytes; 
         nbytes_remain -= tx_bytes; 
      }
      assert(nbytes_remain == 0); 
   }
   if( !m_watchpoints.empty() ) {
      std::map<unsigned,mem_addr_t>::iterator i;
      for( i=m_watchpoints.begin(); i!=m_watchpoints.end(); i++ ) {
         mem_addr_t wa = i->second;
         if( ((addr<=wa) && ((addr+length)>wa)) || ((addr>wa) && (addr < (wa+4))) ) 
            hit_watchpoint(i->first,thd,pI);
      }
   }
}

template<unsigned BSIZE> void memory_space_impl<BSIZE>::read_single_block( mem_addr_t blk_idx, mem_addr_t addr, size_t length, void *data) const
{
   if ((addr + length) > (blk_idx + 1) * BSIZE) {
      printf("GPGPU-Sim PTX: ERROR * access to memory \'%s\' is unaligned : addr=0x%x, length=%zu\n",
             m_name.c_str(), addr, length);
      printf("GPGPU-Sim PTX: (addr+length)=0x%lx > 0x%x=(index+1)*BSIZE, index=0x%x, BSIZE=0x%x\n",
             (addr+length),(blk_idx+1)*BSIZE, blk_idx, BSIZE);
      throw 1;
   }
   typename map_t::const_iterator i = m_data.find(blk_idx);
   if( i == m_data.end() ) {
      for( size_t n=0; n < length; n++ ) 
         ((unsigned char*)data)[n] = (unsigned char) 0;
      //printf("GPGPU-Sim PTX:  WARNING reading %zu bytes from unititialized memory at address 0x%x in space %s\n", length, addr, m_name.c_str() );
   } else {
      unsigned offset = addr & (BSIZE-1);
      unsigned nbytes = length;
      i->second.read(offset,nbytes,(unsigned char*)data);
   }
}

template<unsigned BSIZE> void memory_space_impl<BSIZE>::read( mem_addr_t addr, size_t length, void *data ) const
{
   mem_addr_t index = addr >> m_log2_block_size;
   if ((addr+length) <= (index+1)*BSIZE ) {
      // fast route for intra-block access 
      read_single_block(index, addr, length, data); 
   } else {
      // slow route for inter-block access 
      unsigned nbytes_remain = length;
      unsigned dst_offset = 0; 
      mem_addr_t current_addr = addr; 

      while (nbytes_remain > 0) {
         unsigned offset = current_addr & (BSIZE-1);
         mem_addr_t page = current_addr >> m_log2_block_size; 
         mem_addr_t access_limit = offset + nbytes_remain; 
         if (access_limit > BSIZE) {
            access_limit = BSIZE;
         } 
         
         size_t tx_bytes = access_limit - offset; 
         read_single_block(page, current_addr, tx_bytes, &((unsigned char*)data)[dst_offset]); 

         // advance pointers 
         dst_offset += tx_bytes; 
         current_addr += tx_bytes; 
         nbytes_remain -= tx_bytes; 
      }
      assert(nbytes_remain == 0); 
   }
}

template<unsigned BSIZE> void memory_space_impl<BSIZE>::print( const char *format, FILE *fout ) const
{
   typename map_t::const_iterator i_page;
   for (i_page = m_data.begin(); i_page != m_data.end(); ++i_page) {
      fprintf(fout, "%s - %#x:", m_name.c_str(), i_page->first);
      i_page->second.print(format, fout);
   }
}

template<unsigned BSIZE> void memory_space_impl<BSIZE>::set_watch( addr_t addr, unsigned watchpoint ) 
{
   m_watchpoints[watchpoint]=addr;
}

template class memory_space_impl<32>;
template class memory_space_impl<64>;
template class memory_space_impl<8192>;
template class memory_space_impl<16*1024>;

// union to fuse two 32-bit int into a pointer 
typedef union {
   unsigned char * ptr;
   unsigned int addr[2];
} emu_addr;

std::unordered_map<mem_addr_t, mem_addr_t> memory_space_emu::s_low2hmap;

memory_space_emu::memory_space_emu( std::string name )
{
   m_name = name; 
}

void memory_space_emu::assoc_low_to_haddr( const void *ptr, size_t range )
{
   emu_addr host;
   host.ptr = (unsigned char *)ptr;
   if (host.addr[1] == 0) return; 

   for (size_t offset = 0; offset < range; offset += 0x100000) {
      host.ptr = (unsigned char *)ptr + offset;
      mem_addr_t lowAddr = host.addr[0] & 0xFFF00000; 
      mem_addr_t highAddr = host.addr[1];
      s_low2hmap[lowAddr] = highAddr;
   }
   host.ptr = (unsigned char *)ptr + range - 1;
   mem_addr_t lowAddr = host.addr[0] & 0xFFF00000; 
   mem_addr_t highAddr = host.addr[1];
   s_low2hmap[lowAddr] = highAddr;
}

mem_addr_t memory_space_emu::xlate_low_to_haddr( mem_addr_t lowAddr ) 
{
   std::unordered_map<mem_addr_t, mem_addr_t>::const_iterator i_addr;
   i_addr = s_low2hmap.find(lowAddr & 0xFFF00000);
   if (i_addr != s_low2hmap.end()) {
      return i_addr->second;
   } else {
      return ((lowAddr & 0x80000000)? 0x7fff : 0);
   }
}

void memory_space_emu::dump_low_to_haddr(FILE *fout)
{
   std::unordered_map<mem_addr_t, mem_addr_t>::const_iterator i_addr;
   for (i_addr = s_low2hmap.begin(); i_addr != s_low2hmap.end(); ++i_addr) {
      fprintf(fout, "[%#08x] = %#08x", i_addr->first, i_addr->second);
   }
   fflush(fout);
}

void g_dump_low_to_haddr(FILE *fout)
{
   memory_space_emu::dump_low_to_haddr(fout);
}

unsigned char * memory_space_emu::emu_addr_to_host( mem_addr_t addr ) const 
{
   // assume memory address above 0xC0000000 are in the host's stack
   emu_addr host;
   host.addr[0] = addr; 
   // host.addr[1] = xlate_low_to_haddr(addr);
   host.addr[1] = (addr & 0x80000000)? 0x7fff : 0x0; // assume that all addresses are in the lower 2GB

   return host.ptr;
}

void memory_space_emu::write( mem_addr_t addr, size_t length, const void *data, ptx_thread_info *thd, const ptx_instruction *pI )
{
   unsigned char *dst = emu_addr_to_host(addr);
   memcpy(dst, (const unsigned char*)data, length);
}

void memory_space_emu::read( mem_addr_t addr, size_t length, void *data ) const 
{
   unsigned char *src = emu_addr_to_host(addr);
   memcpy((unsigned char *)data, src, length);
}

void memory_space_emu::print( const char *format, FILE *fout ) const
{
   // nothing to print as it does not contain any real data 
}

void g_print_memory_space(memory_space *mem, const char *format = "%08x", FILE *fout = stdout) 
{
   mem->print(format,fout);
}

#ifdef UNIT_TEST

int memory_space_impl_test(int argc, char *argv[] )
{
   int errors_found=0;
   memory_space *mem = new memory_space_impl<32>("test",4);
   // write address to [address]
   for( mem_addr_t addr=0; addr < 16*1024; addr+=4) 
      mem->write(addr,4,&addr,NULL,NULL);

   for( mem_addr_t addr=0; addr < 16*1024; addr+=4) {
      unsigned tmp=0;
      mem->read(addr,4,&tmp);
      if( tmp != addr ) {
         errors_found=1;
         printf("ERROR ** mem[0x%x] = 0x%x, expected 0x%x\n", addr, tmp, addr );
      }
   }

   for( mem_addr_t addr=0; addr < 16*1024; addr+=1) {
      unsigned char val = (addr + 128) % 256;
      mem->write(addr,1,&val,NULL,NULL);
   }

   for( mem_addr_t addr=0; addr < 16*1024; addr+=1) {
      unsigned tmp=0;
      mem->read(addr,1,&tmp);
      unsigned char val = (addr + 128) % 256;
      if( tmp != val ) {
         errors_found=1;
         printf("ERROR ** mem[0x%x] = 0x%x, expected 0x%x\n", addr, tmp, (unsigned)val );
      }
   }

   if( errors_found ) {
      printf("SUMMARY:  ERRORS FOUND\n");
   } else {
      printf("SUMMARY: UNIT TEST PASSED\n");
   }
}

int memory_space_emu_test(int argc, char *argv[] )
{
   // initialize two memory buffers one in the stage, one malloc'ed
   unsigned char stack_buffer[100];
   unsigned int *heap_buffer = new unsigned int[100];

   for (int t = 0; t < 100; t++) {
      stack_buffer[t] = 20 + t;
      heap_buffer[t] = 500 - t * 3;
   }

   printf("stack_buffer = %p\n", stack_buffer);
   printf("heap_buffer = %p\n", heap_buffer);

   int errors_found=0;
   memory_space *mem = new memory_space_emu("test");

   emu_addr stack_addr;
   stack_addr.ptr = stack_buffer;
   mem_addr_t m_stack = stack_addr.addr[0];

   emu_addr heap_addr;
   heap_addr.ptr = reinterpret_cast<unsigned char*>(heap_buffer);
   mem_addr_t m_heap = heap_addr.addr[0];

   // copy half of stack to heap
   for (int t = 0; t < 50; t++) {
      unsigned char src; 
      mem->read(m_stack + t, 1, &src);
      if (src != 20 + t) errors_found = 1;

      unsigned int dst = src;
      mem->write(m_heap + t * sizeof(unsigned int), sizeof(unsigned int), &dst);
      if (heap_buffer[t] != 20 + t) errors_found = 1;
   }

   // copy half of stack to heap
   for (int t = 50; t < 100; t++) {
      unsigned int src; 
      mem->read(m_heap + t * sizeof(unsigned int), sizeof(unsigned int), &src);
      if (src != 500 - t * 3) errors_found = 1;

      unsigned char dst = src;
      mem->write(m_stack + t, 1, &dst);
      if (stack_buffer[t] != ((500 - t * 3) % 256)) errors_found = 1;
   }

   if( errors_found ) {
      printf("SUMMARY:  ERRORS FOUND\n");
   } else {
      printf("SUMMARY: UNIT TEST PASSED\n");
   }
}

int main(int argc, char *argv[] )
{
   memory_space_impl_test(argc, argv);
   memory_space_emu_test(argc, argv);
}

#endif
