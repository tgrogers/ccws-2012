//------------------------------------------------------------------------------
// (c) tgrogers - UBC 2011
//------------------------------------------------------------------------------

#ifndef __GPGPUSIM_CL_CUSTOM_INTERFACE_FUNCTIONS_H__
#define __GPGPUSIM_CL_CUSTOM_INTERFACE_FUNCTIONS_H__

#include <CL/cl.h>
#include <boost/dynamic_bitset.hpp>

extern CL_API_ENTRY void CL_API_CALL
virtualMemoryDeclaration(  cl_command_queue commandQueue,
                           void* start_of_vm,
                           size_t total_size,
                           size_t field_size,
                           size_t tile_width,
                           size_t object_size,
                           size_t page_size,
                           boost::dynamic_bitset<> hot_bitset );
#endif
