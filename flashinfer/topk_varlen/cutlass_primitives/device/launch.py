"""Programmatic dependent launch (PDL): overlap this kernel's prologue with the previous kernel.

With PDL enabled on the launch, the driver may start this grid's CTAs while the previous
kernel in the stream is still finishing.  The CTAs run their prologue (index math, shared
memory setup) and then block in ``wait_for_prior_grid`` until the previous kernel has
completed and flushed its memory.  Near the end, ``release_dependent_grid`` lets the *next*
kernel's CTAs start early in turn.  Measured on the streaming kernel: 0.4 to 0.6 us hidden per
launch on most cells.

Placement rules learned by measurement:

* ``wait_for_prior_grid`` goes before the first global read.  Everything read from global
  memory (inputs, lengths, this kernel's own self-resetting counters from its previous
  launch) may still be in flight in the previous kernel.
* ``release_dependent_grid`` goes at the very end.  Releasing after the main pass let the
  next launch's CTAs take SM slots during the epilogue: +2 to 3% at 256K rows, batch 64.

Both are SM90 or newer.  Kernels for older parts must not trace them; the configuration's
``pdl`` field is chosen at Python level from the capability.
"""

import cutlass.cute as cute

__all__ = ["wait_for_prior_grid", "release_dependent_grid"]


@cute.jit
def wait_for_prior_grid():
    """Block until the previous kernel in the stream has completed (``griddepcontrol.wait``).

    A no-op when the launch was not programmatic.  Costs nothing when the prior grid already
    finished.
    """
    cute.arch.griddepcontrol_wait()


@cute.jit
def release_dependent_grid():
    """Allow the next kernel in the stream to begin launching
    (``griddepcontrol.launch_dependents``).  Once per CTA; issue at the end of the kernel."""
    cute.arch.griddepcontrol_launch_dependents()
