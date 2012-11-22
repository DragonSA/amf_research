# Makefile for figures
#
# FIGURES	List of figures that need to be built

SRC=		../src/

all: ${FIGURES}

clean:
	@echo "[CLEAN] common"
	@rm -f fig_*.pdf fig_*.eps

.PHONY:		all clean
.SUFFIXES:	.py .pdf .eps
.PATH.py:	${SRC}

.include "${SRC}Makefile.incl"
