PAPER = flashspec

.PHONY: all clean

all:
	pdflatex $(PAPER).tex
	bibtex   $(PAPER)
	pdflatex $(PAPER).tex
	pdflatex $(PAPER).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.pdf
