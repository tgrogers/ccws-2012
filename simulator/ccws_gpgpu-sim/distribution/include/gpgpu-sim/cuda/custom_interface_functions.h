//------------------------------------------------------------------------------
// (c) tgrogers - UBC 2011
//------------------------------------------------------------------------------

#ifndef __GPGPUSIM_CUDA_CUSTOM_INTERFACE_FUNCTIONS_H__
#define __GPGPUSIM_CUDA_CUSTOM_INTERFACE_FUNCTIONS_H__

#include <boost/dynamic_bitset.hpp>

void virtualMemoryDeclaration( void* start_of_vm,
                           size_t total_size,
                           size_t field_size,
                           size_t tile_width,
                           size_t object_size,
                           size_t page_size,
                           boost::dynamic_bitset<> hotbytes);

#endif
