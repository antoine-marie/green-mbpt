/*
 * Copyright (c) 2023 University of Michigan
 *
 */
#ifndef MBPT_COMMON_UTILS_H
#define MBPT_COMMON_UTILS_H

#include <green/utils/timing.h>
#include <green/grids.h>
#include "df_integral_t.h"

#include "common_defs.h"

namespace green::mbpt {

  inline void print_leakage(double leakage, const std::string& object) {
    std::cout << "Leakage of " + object << ": " << leakage << std::endl;
    if (leakage > 1e-8) std::cerr << "WARNING: The leakage is larger than 1e-8" << std::endl;
  }

  std::array<double, 3> compute_energy(const ztensor<5>& g_tau, const ztensor<4>& sigma1, const ztensor<5>& sigma_tau,
                                       const ztensor<4>& H_k, const grids::transformer_t& ft,
                                       const symmetry::brillouin_zone_utils& bz, bool X2C);

  inline std::pair<size_t, size_t> compute_local_and_offset_node_comm(size_t size, const utils::mpi_context & cntx = utils::mpi_context::context()) {
    size_t local = size / cntx.node_size;
    local += (size % cntx.node_size > cntx.node_rank) ? 1 : 0;
    size_t offset = local * cntx.node_rank +
                        ((size % cntx.node_size > cntx.node_rank) ? 0 : (size % cntx.node_size));
    return {local, offset};
  }

  using G_type     = utils::shared_object<ztensor<5>>;
  using S1_type    = ztensor<4>;
  using St_type    = utils::shared_object<ztensor<5>>;

  template <typename prec>
  void valence_slice_matrix_inplace(size_t nao, size_t ncore, size_t nv, const std::vector<int>& core_reordering, MatrixX<prec>& M) {
	for (size_t r = 0; r < nv; ++r) {
	    for (size_t s = 0; s < nv; ++s) {
		    M(r, s) = M(core_reordering[ncore + r], core_reordering[ncore + s]);
	    }
	  }
  }

  template <typename prec>
  void valence_slice_coulint_inplace(size_t nao, size_t ncore, size_t nv, size_t NQ, const std::vector<int>& core_reordering, tensor<prec, 3>& v) {
  // Slice in place: compact the valence (r, s) block for each iq into the
  // front of the same buffer used for v, instead of allocating v_val.
  // Since we only ever write to earlier (iq, r, s) than we read from
  // (compaction moves data "down"), no aliasing/overwrite issue occurs
  // as long as we iterate iq, r, s in increasing order. 
  for (size_t iq = 0; iq < NQ; ++iq) {
    for (size_t r = 0; r < nv; ++r) {
	    for (size_t s = 0; s < nv; ++s) {
		    v(iq, r, s) = v(iq, core_reordering[ncore + r], core_reordering[ncore + s]);
	    }
	  }
  }
}

}  // namespace green::mbpt
#endif  // MBPT_COMMON_UTILS_H
