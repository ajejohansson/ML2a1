# ML2a1
Andreas Johansson submission for LT2326, assignment 1

Brief script procedure instructions:

1 Run preprocess.py, arg 1: optional, specifying language-level subset -> generates pickled dataset splits (one pytorch Dataset subset per train/test/val)
2 Run train.py without arguments -> trains on currently saved trainset, saves model
3 Run test.py, arg1: mandatory, specifying eval set, arg 2: optional, for writing to file instead of printing -> evaluates currently saved model on currently saved dataset

preprocess should not be run again between train and test. config.json should not be changed until after eval if preprocess was run without an argument (eval results will be correct but will be associated with the wrong settings)

Details/design decisions:
(focused on the preprocess part, since there's where most design decisions wound up)
1 preprocess.py

    Creates pickled train/test/val instances of the ThaiOCRData class (subclass of torch Dataset)
    If run with a command line argument, this argument should either
        specify language/subset, e.g. 'thai', 'english', 'numeric' (case insensitive), in which case the generated datasets will include all subsets of that subset, or
        be literally anything (or 'both' or 'thai & english' to be to be semantically appropriate if syntactically unnecessary), in which case the datasets will be based on everything in both languages
    If run without an argument, the script will use the json configuration file. The file is largely self-explanatory with the following clarifications:
        The values of the "lang" keys follows the same rules as for command line arguments; if they are direct daughters of 'ThaiOCR-TrainigSet' (e.g. 'thai', 'numeric', case insensitive), this subset will be used, otherwise English + Thai will be used.
        Unlike running with command line arguments, the config file is used later for evaluation printing, so while the "lang" values can be "sdajdhasjhdajks" and be semantically equivalent to "Thai & English", the latter is preferred for clarity. Both "lang" values should be the same (only matters in some configs, but best to always do it)
        Config "dpi" and "typeface" must be lists (arrays?) of strings even if only one subset is used: [ "200" ] and [ "200", "300", "400" ] are valid; "200" is not.
        If a config is given such that it does not pick out any pictures (such as by giving no valid dpis or typefaces in their respective arrays), sklearn's split function will complain that there aren't any samples.

    Behaves differently depending on whether the test and eval data are disjoint: if the two are categorically different, there is no need to split the data further; i.e. 'Thai normal text, 400 dpi' and 'Thai bold text, 400dpi' have no data overlap since all it takes is for a single variable (typeface here) to differ for the entire sample to be disjoint. Some subsets get quite small already, and this behaviour avoids splitting more than necessary. Note that the code is not robust to all kinds of configurations, but it does work on the experiments. For example, one could feasible want to train on thai + eng but evaluate only on thai (so eng is basically noise), but the current setup assumes train and eval will use the same language. There are also ways to trick the function into interpreting the data split as disjoint when it is not (leading to dataset contamination), but as far as I can tell, this is not the case with the current experiment splits.

    Note that, in the case of a typical train/test/val split, the split is made per-language before concatenation. The upside of split->concat (as opposed to the other way around) is that the datasets then have a controlled subset balance. The downside is that the datasets technically aren't fully randomly sampled. This is only for language/special/numeric; it is still fully random with regards to dpi and typeface. 

    The notable amount of conditional preprocessing is partially to deal deal with the somewhat intricate splitting process above, but also in order to avoid computation later. When the data is filtered at the very first step, that means future iterations are only over relevant data, and no expensive operations are done on data that will not be used in a given training/eval sequence.

    The ThaiOCRData class itself has fewer design decisions compared to the preprocessing, but one is the use of transforms.Resize((72, 48))
    The data needed to be resized, but this specific size is largely arbitrary (it intuitively matches an average size ratio). I considered getting the average size in both dimension and using that (rounded to something nice), but I don't have a concept for if that would be useful. This size does seem to handle most experiments well enough, including when training and testing on different dpi.

2 train.py
    Trains a model on the currently saved training dataset pickle. Uses hyperparameters per model class definition and (currently) 4 epochs.
    Another config file for model hyperparams could have quite simply been made, but I did not because 1. the model performed relatively well without too much tweaking, so there was less use of flexibly changing hyperparams, and 2. adding hyperparam tweaking would have exploded the number of possible experiments/number of times the model needs to be run, which could rack up a lot of time given the time it can take to train neural models.

3 test.py
    Evaluates the currently saved model on a specified evaluation set
    If arg 1 is 'test', evaluates on test set. If it is 'val', evaluates on val set. If it is anything else, defaults to val set regardless, but with a message that it did not recognise the argument. If there is no arg 2, prints the evaluation metrics. If there is an arg 2 (any argument but 'filewrite' could be sensible), instead writes the results to a new file in the results directory.

3 Run test.py, arg1: mandatory, specifying eval set, arg 2: optional, for writing to file instead of printing -> evaluates currently saved model on a currently saved evalset

Analysis

Since there are quite a few tests and a lot of classes, this analysis will be fairly broad strokes.
The results files have generic names but the languages (if on all data) or config file (if not) are specified at the top of each file.
Mostly trained on 4 epochs unless otherwise specified (an epoch datapoint has been manually added to a few files, not part of the script), in order to cut down on training time.
All logs are on their respective test set, and for each of them it is the first time any model has been run on the test set. Any hyperparameter experimentation was only on the val set. 
Rounding to whole percentages.

First off, it is clear that the model benefits from more epochs than my most common 4; logs 1, 2 and 13 (both languages, all subsets) are identical (with set random seeds both for splitting and pytorch, assuming I did it right) except for the number of epochs, with 8 epochs taking the accuracy from 74% to 83%, and 12 epochs increasing only some tenths of a percent. Similarly, logs 10,11,12 (specifics later, since it's my added experiment) only differ in epochs, with 67% at 4 epochs, 82% at 8, but 72% at 12. So while 4 seems too few, at 12 there might be some overfitting. With this in mind, the rest of the experiment results could likely be improved with more epochs.

The model performs similarly well on both languages 84 (eng) vs 82% (thai) accuracy, with relatively matching macro recalls/precisions/f-scores. The difference between the languages could simply be that Thai has more classes to learn (98 vs 52). The Eng+Thai evaluation is notably lower (74) at the same number of epochs, but as suggested earlier, but since 8 epochs brings this number up to 83, it might just be that the 150 classes (and language differences) might just require more training to learn. The rest of the experiments (barring the one I added logs 10,11,12) range between 71 and 82. The lowest are the mismatched results (logs 6,7,8) at 71-72%, but it seems likely these patterns mostly requires more training given the later results of logs 10, 11, 12 (next paragraph). The non-accuracy metrics do not overly suffer with mismatched evaluation data (there are some cases, but these exist also in the matched evaluation data).

My added experiment trains on English, 200dpi, bold and evaluates on English, 300 and 400dpi, normal. It is more adversarial than the other experiments since it is mismatched both in dpi and typeface, and the model performs worse on it than any other experiment (67% accuracy), presumably as a result. Since 8 epochs brings the performance higher than any other mismatched evaluation by almost 10 percentage points, it seems the adversarial nature of the experiment is not something the model cannot overcome.

For some qualitative points (using results log11: the best performance on my added experiment):
    The model has some issues with characters whose capital and lower case counterparts look alike apart from size.
        E.g. 'w' vs 'W' and 'p' vs 'P': both recall and precision for the lower case counterpart is 0, whereas the capital versions perform much better. It might be that the model vastly favours the more common counterpart in these and similar cases.
        On the one hand, these are likely more difficult to tell apart than the average characters, so some performance issues relative to other classes should perhaps be expected. However, this is likely exacerbated by the resizing I use for the images, which likely deletes the largely size-based distinction exists between these characters. I use pytorch's default bilinear interpolation resizing mode since I lacked the concept of what would be better, but I could see resizing via padding working better for these characters. I somewhat doubt this approach is better overall, however, but some more sophisticated resizing could likely work best.

        Similar cases occur across the log. Sometimes both capital and lower case character performs well, but it is rare to find a low-performing case whose counterpart does not perform at least relatively well. This observation of course also extends across characters that have similar-looking characters that are not technically its upper- or lower-case counterpart.

    Interestingly, the 12-epoch counterpart (log12) does not have a single 0 across its per-class metrics, so while its accuracy is lower (suggesting overfitting), it is not worse across the board.

    Finally: not surprisingly, italic characters seem more adversarial. All configs being the same except the evaluation typeface being set to 'italic' results in 51% accuracy (log14). Running the same training setup but for 8 epochs (log15) reduces this to 36%, suggesting it certainly is not something to be solved with more training: it requires some more fundamental change to model architecture.
    (Note that this is speculation based on one training per epoch number, so not very empirically sound. But even so.)

