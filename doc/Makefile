# Note: Tabs must be used instead of spaces
#       Using spaces results in a "missing separator" error
filename1 = problem-statement
pdf: clean
	pdflatex $(filename1)   # Creates a bad pdf and lots of temp files
    #	bibtex $(filename1)     # Creates filename.bbl, needed by latex
	pdflatex $(filename1)   # Produces a slightly better pdf and temp files
	pdflatex $(filename1)   # Produces a correct pdf from the temp files

	make clean
clean:
	rm -f $(filename1).aux $(filename1).log $(filename1).out $(filename1).spl
	rm -f $(filename1).blg $(filename1).bbl
