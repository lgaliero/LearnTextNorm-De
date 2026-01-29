from optparse import OptionParser

def evaluate(raw, gold, pred, ignCaps=False, verbose=False):
    cor = 0
    changed = 0
    total = 0

    if len(gold) != len(pred):
        err('Error: gold normalization contains a different numer of sentences(' + str(len(gold)) + ') compared to system output(' + str(len(pred)) + ')')

    for sentRaw, sentGold, sentPred in zip(raw, gold, pred):
        if len(sentGold) != len(sentPred):
            err('Error: a sentence has a different length in you output, check the order of the sentences')
        for wordRaw, wordGold, wordPred in zip(sentRaw, sentGold, sentPred):
            if ignCaps:
                wordRaw = wordRaw.lower()
                wordGold = wordGold.lower()
                wordPred = wordPred.lower()
            if wordRaw != wordGold:
                changed += 1
            if wordGold == wordPred:
                cor += 1
            elif verbose:
                print(wordRaw, wordGold, wordPred)
            total += 1

    accuracy = cor / total
    lai = (total - changed) / total
    error_reduction = (accuracy - lai) / (1-lai)  # ← Changed from 'err' to 'error_reduction'

    print('Baseline acc.(LAI): {:.2f}'.format(lai * 100)) 
    print('Accuracy:           {:.2f}'.format(accuracy * 100)) 
    print('ERR:                {:.2f}'.format(error_reduction * 100))  # ← Changed here too

    return lai, accuracy, error_reduction  # ← And here

def loadNormData(path):
    raw = []
    gold = []
    with open(path, 'r', encoding='utf-8') as f:
        rawSent = []
        goldSent = []
        for line in f:
            line = line.rstrip('\n\r')
            
            if not line.strip():  # Blank line
                if rawSent and goldSent:
                    raw.append(rawSent)
                    gold.append(goldSent)
                rawSent = []
                goldSent = []
                continue
            
            parts = line.split('\t')
            if len(parts) >= 2 and parts[0].strip() and parts[1].strip():  # ← ADD THIS CHECK
                rawSent.append(parts[0].strip())
                goldSent.append(parts[1].strip())
        
        if rawSent and goldSent:
            raw.append(rawSent)
            gold.append(goldSent)
    
    return raw, gold

def err(msg):
    print('Error: ' + msg)
    exit(0)

if __name__ == '__main__':
    parser = OptionParser(description='Normalization baselines')
    parser.add_option("--gold", help='path to the gold normalization data')
    parser.add_option("--pred", help='path to the system output normalization')
    parser.add_option("--ignCaps", action='store_true', default=False, help='lowercase everything')
    parser.add_option("--verbose", action='store_true',  default=False, help='print the errors made by the system')

    (opts, args) = parser.parse_args()
    if opts.gold == None:
        err('Please provide gold data with --gold')
    if opts.pred == None:
        err('Please provide system output with --pred')
    
    goldRaw, goldNorm = loadNormData(opts.gold)
    predRaw, predNorm = loadNormData(opts.pred)
    evaluate(goldRaw, goldNorm, predNorm, opts.ignCaps, opts.verbose)

