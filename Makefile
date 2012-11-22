DIRS=		presentation proposal report

.PHONY: 	all clean test figures ${DIRS}

all: ${DIRS}

clean:
	@${MAKE} -C common clean
.for dir in ${DIRS}
	@${MAKE} -C ${dir} -f ../Makefile.incl clean NAME=${dir}
.endfor

test:
	${PYTHON} -m unittest discover src/

figures:
	@${MAKE} -C common FIGURES="`(cd src; ls fig_*.py) | sed -e 's|\\.py$$|.pdf|g'`"

.for dir in ${DIRS}
${dir}:
	@${MAKE} -C ${dir} -f ../Makefile.incl NAME=${dir}
.endfor

.include "commands.mk"
