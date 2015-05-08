import csv

def textAsString(filename, axis, splitter=' '):
    """Read from a text file and parse each train data as a list of string

    parameters
    ----------
    filename:  path of the file
    axis:      indicates which feature is the label, while others are datas
               if axis is 0, no labels
    Splitter:  Split each line, ex., ' ' or '\t'

    Returns
    -------
    X:  a list of features of the data
    y:  a list of labels of the data, [] if does not exist
    """
    X = []; y = []
    f = open(filename, 'rU')
    for line in f:
        line = line.rstrip('\n')
        data = line.split(splitter)
        if axis == None:
            X.append(data)
        else:
            label = [data.pop(axis)]
            y.append(label)
            X.append(data)
    f.close
    if axis == None:
        return X
    else:
        return X, y

def textAsFloat(filename, axis, splitter=' '):
    """Read from a text file and parse each train data as a list of float data

    parameters
    ----------
    filename:  path of the file
    axis:      indicates which feature is the label, while others are datas
               if axis is 0, no labels
    Splitter:  Split each line, ex., ' ' or '\t'

    Returns
    -------
    X:  a list of features of the data
    y:  a list of labels of the data, [] if does not exist
    """
    X = []; y = []
    f = open(filename, 'rU')
    for line in f:
        line = line.rstrip('\n')
        data = map(float, line.split(splitter))
        if axis == None:
            X.append(data)
        else:
            label = [data.pop(axis)]
            y.append(label)
            X.append(data)
    f.close
    if axis==None:
        return X
    else:
        return X,y

def csvAsString(filename, yaxis=None):
    with open(filename, 'rU') as f:
        reader = csv.reader(f)
        idList = []; X = []; y = [];
        reader.next()
        for line in reader:
            if yaxis==None:
                idList.append(int(line[0]))
                X.append(line[1:])
            else:
                idList.append(int(line[0]))
                X.append(line[1:-1])
                y.append(float(line[yaxis]))
    if yaxis==None:
        return X
    else:
        return X,y