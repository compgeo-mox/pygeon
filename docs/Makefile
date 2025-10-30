# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
APIDOC        = sphinx-apidoc

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile apidoc reports clean strict

# Auto-generate API documentation
apidoc:
	@echo "Auto-generating API documentation..."
	$(APIDOC) -f -o api ../src/pygeon --separate --no-toc
	@echo "API documentation generated in api/"

# Collect/generated reports
reports:
	@echo "Collecting generated reports..."
	bash readthedocs/hooks/pre_build_reports.sh
	@echo "Reports collected into reports/"

# Clean build artifacts and generated API docs
clean:
	rm -rf $(BUILDDIR)
	@echo "Cleaned build directory"

# Strict build (same as CI) - warnings become errors
strict: reports apidoc
	@echo "Building with strict checks (warnings = errors)..."
	$(SPHINXBUILD) -W -n --keep-going -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "✅ Strict build passed! Your docs will pass CI."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile reports apidoc
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
