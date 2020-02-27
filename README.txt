There are 2 possible versions of each file in this dataset.

1) file.pos -- there are two columns separated by a tab:
   1st column: token
   2nd column: POS tag
   Blank lines separate sentences.

   This is the format of training files, system output, and development
   or test files used for scoring purposes.

2) file.words -- one token per line, with blank lines between sentences.
   Format of an input file for a tagging program.

For this assignment, we are distributing the following files:

POS_train.pos  -- to use as the training corpus

POS_dev.words   -- to use as your development set (for testing your system)

POS_dev.pos     -- to use to check how well your system is doing

POS_test.words -- to run your system on.  You should produce a file in
	     	the .pos format as your output and submit it as per the
		submission instructions to be announced.

scorer.py -- the scorer we develop to evaluate the output files