def GetScoreSepsis(confusionMatrix):
    '''
     Funkce pro vyhodnocení úspěšnosti modelu
    :param confusionMatrix: Matice záměn z funkce Main()
    :return: Výsledná senzitivita modelu, sp: Výsledná specificita modelu
    acc: Výsledná přesnost modelu
    fScore: Výsledné F1 skóre modelu
    ppv: Pozitivní prediktivní hodnota
    '''


    tn = confusionMatrix[1, 1]
    tp = confusionMatrix[2, 2]
    fp = confusionMatrix[2, 1]
    fn = confusionMatrix[1, 2]

    if (tp + fn) == 0:
        se = NaN
    else:
        se = tp / (tp + fn)


    if (tn + fp) == 0:
        sp = NaN
    else:
        sp = tn / (tn + fp)

    if (tp + fp) == 0:
        ppv = NaN
    else:
        ppv = tp / (tp + fp)


    if (tp + tn + fp + fn) == 0:
        acc = NaN
    else:
        acc = (tp + tn) / (tp + tn + fp + fn)


    if (tp + fn + fp) == 0:
        fScore = NaN
    else:
        fScore = (2 * tp) / (2 * tp + fn + fp)
