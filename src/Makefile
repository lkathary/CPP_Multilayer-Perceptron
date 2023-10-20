.PHONY: tests build mlp
CXX=g++
CAR=ar
CRANLIB=ranlib

OS=$(shell uname)

FLAGS=-Wall -Wextra -std=c++17
# FLAGS=-Wall -Werror -Wextra -std=c++17

GTEST=-lgtest_main -lgtest -lpthread
GCOV=-fprofile-arcs -ftest-coverage

TARGETDIR=./
REPORTDIR=gcovdir/
BDIR=../build/
INSTALL_DIR = ../MLP_app
INSTALL_DIR_MACOS = MLP.app/Contents/MacOS


FILE=Mlp

FILE_MATRIX=matrix
FILE_NET=network
FILE_MATRIX_NET=matrixnetwork
FILE_GRAPH_NET=graphnetwork
FILE_TEST=test_mlp

all: mlp

mlp: build
ifeq ($(OS), Linux)
	$(BDIR)/$(FILE)
else
	open $(BDIR)/$(FILE).app
endif

build: clean
	-mkdir $(BDIR)
	cp $(FILE).pro $(BDIR)
	cp *.h *.cpp *.ui $(BDIR)
	cd $(BDIR); qmake $(FILE).pro
	make -C $(BDIR)

tests:
	$(CXX) -c $(FLAGS) $(TARGETDIR)$(FILE_MATRIX).cpp
	$(CXX) -c $(FLAGS) $(TARGETDIR)$(FILE_NET).cpp
	$(CXX) -c $(FLAGS) $(TARGETDIR)$(FILE_MATRIX_NET).cpp
	$(CXX) -c $(FLAGS) $(TARGETDIR)$(FILE_GRAPH_NET).cpp
	$(CXX) -c $(FLAGS) $(FILE_TEST).cpp $(GTEST)
	$(CXX) -o $(TARGETDIR)$(FILE_TEST) $(FLAGS)\
	          $(FILE_TEST).o $(FILE_MATRIX).o $(FILE_NET).o $(FILE_MATRIX_NET).o $(FILE_GRAPH_NET).o -L $(GTEST)
	-$(TARGETDIR)$(FILE_TEST)

gcov_report: clean
	$(CXX) -c $(FLAGS) $(TARGETDIR)$(FILE_MATRIX).cpp $(GCOV)
	$(CXX) -c $(FLAGS) $(TARGETDIR)$(FILE_NET).cpp $(GCOV)
	$(CXX) -c $(FLAGS) $(TARGETDIR)$(FILE_MATRIX_NET).cpp
	$(CXX) -c $(FLAGS) $(TARGETDIR)$(FILE_GRAPH_NET).cpp
	$(CXX) -c $(FLAGS) $(FILE_TEST).cpp $(GTEST) $(GCOV)
	$(CXX) -o $(TARGETDIR)$(FILE_TEST) $(FLAGS)\
	          $(FILE_TEST).o $(FILE_MATRIX).o $(FILE_NET).o $(FILE_MATRIX_NET).o $(FILE_GRAPH_NET).o $(GCOV) -L $(GTEST)
	-$(TARGETDIR)$(FILE_TEST)

	gcov *.cpp
	lcov -b ./ -d ./ --gcov-tool /usr/bin/gcov -c -o $(FILE).info
	lcov --remove $(FILE).info '/usr/*' -o $(FILE)_filtered.info
	genhtml -o $(REPORTDIR) -t "Unit-tests of $(FILE)" $(FILE)_filtered.info
ifeq ($(OS), Linux)
	xdg-open $(REPORTDIR)index.html
else
	open $(REPORTDIR)index.html
endif

install: build
	rm -rf $(INSTALL_DIR)
	mkdir $(INSTALL_DIR)
ifeq ($(OS), Linux)
	cp $(BDIR)$(FILE) $(INSTALL_DIR)
	-cp -R datasets $(INSTALL_DIR)/datasets/
	-cp -R images $(INSTALL_DIR)/images/
	-cp -R weights $(INSTALL_DIR)/weights/
else
	cp -r $(BDIR)/$(FILE).app $(INSTALL_DIR)
	-cp -R datasets $(INSTALL_DIR)/datasets/
	-cp -R images $(INSTALL_DIR)/images/
	-cp -R weights $(INSTALL_DIR)/weights/
endif
	make clean
	echo "Installed to" $(INSTALL_DIR)

uninstall:
	rm -rf $(INSTALL_DIR)
	echo "Unistalled from" $(INSTALL_DIR)

dvi:
ifeq ($(OS), Linux)
	echo "Creating DVI file..." && \
	texi2dvi $(TARGETDIR)readme.tex && \
	echo "Create PDF file..." && \
	texi2pdf $(TARGETDIR)readme.tex && \
	echo "Done" || echo "\nSomething went wrong. Using the pre-built .pdf file"
	xdg-open $(TARGETDIR)readme.pdf
else
	echo "Using the pre-built .pdf file"
	open $(TARGETDIR)readme.pdf
endif

dist:
	rm -rf mlp_dist/
	mkdir mlp_dist/
	mkdir mlp_dist/src
	cp Makefile Mlp.pro mlp_dist/src/
	cp *.cpp *.h *.ui mlp_dist/src/
	-cp readme.tex readme.pdf mlp_dist/src/
	-cp -R datasets mlp_dist/src/
	-cp -R images mlp_dist/src/
	-cp -R weights mlp_dist/src/
	tar cvzf ../mlp_dist.tgz mlp_dist/
	rm -rf mlp_dist/
	
prereq:
ifeq ($(OS), Linux)
	sudo apt update
	sudo apt-get install qtchooser
	sudo apt-get install qt5-default

	sudo apt-get install texlive
	sudo apt-get install texinfo
#	sudo apt-get install lcov
else
	echo "Please install the necessary modules if you have enough disk space"

#	brew install texlive
#	brew install lcov
endif


check:
	-rm -rf cpplint.py CPPLINT.cfg
	-ln -s ../materials/linters/cpplint.py
	-ln -s ../materials/linters/CPPLINT.cfg
	-python3 cpplint.py --extensions=cpp *.cpp *.h
	cppcheck --enable=all --check-config --suppress=missingIncludeSystem --language=c++ *.cpp *.h

leaks: tests
ifeq ($(OS), Linux)
# 	Clear the log
#	valgrind --leak-check=full ./
	echo "Leaks check" > __leaks_check.out

	-CK_FORK=no valgrind --leak-check=full $(TARGETDIR)$(FILE_TEST) 2>> __leaks_check.out
	-grep -e leaks __leaks_check.out
	-grep -e error __leaks_check.out
else
	# Comment-out the 'ASSERT_THROW()' tests to fix false positives
	CK_FORK=no leaks -atExit -- $(TARGETDIR)$(FILE_TEST)
endif

clean:
	rm -rf $(BDIR)
	rm -rf *.gcno
	rm -rf *.gcda
	rm -rf *.gcov
	rm -rf *.info
	rm -rf $(REPORTDIR)
	rm -rf  *.o *.a *.out
	rm -rf $(TARGETDIR)$(FILE_TEST)
	rm -rf CPPLINT.cfg cpplint.py
	rm -rf *.exe *.user
	rm -rf *.dvi *.log *.aux

# Frequent typos/synonyms
claen:	clean
clear:	clean
claer:	clean

rebuild:
	make clean
	make all
	rm -rf  *.o *.a
