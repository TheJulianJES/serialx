"""Package setup configuration."""

import sys

import setuptools

if __name__ == "__main__":
    # Keep in sync with the `setuptools-rust` marker in `build-system.requires`
    if sys.platform in ("win32", "darwin"):
        from setuptools_rust import Binding, RustExtension

        setuptools.setup(
            rust_extensions=[
                RustExtension(
                    "serialx._serialx_rust",
                    binding=Binding.PyO3,
                    optional=True,
                )
            ]
        )
    else:
        setuptools.setup()
