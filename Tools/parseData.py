
def textAsMat(filename, yColNum, splitter=' '):
    """Read from a text file and parse as a list
    read each line as a train data
    Split each line by a 'splitter', like ' ' or '\t'
    label is the 'yColNum' column, others are datas
    if 'yColNum' is 0, no labels
    """
    X = []; y = []
    f = open(filename, 'rU')
    for line in f:
        line = line.rstrip('\n')
        data = map(float, line.split(splitter))
        if yColNum == 0:
            X.append(data)
        else:
            label = [data.pop(yColNum-1)]
            y.append(label)
            X.append(data)
    f.close
    return X, y