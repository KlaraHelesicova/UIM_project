def main(filepath):

'''
# Funkce slouží pro ověření klasifikačních schopností navrženého modelu.
#Model bude ověřován na skryté množině dat, v odevzdaném projektu je proto nutné dodržet sktrukturu tohoto kódu

Vstup: filepath: Název složky (textový řetězec) obsahující data
Výstup:    se:                 Výsledná senzitivita modelu
    sp:                 Výsledná specificita modelu
        acc:                Výsledná přesnost modelu
       fScore:             Výsledné F1 skóre modelu
        ppv:                Pozitivní prediktivní hodnota
        confusionMatrix:    Matice záměn

Funkce:    MyModel()           Funkce pro implementaci modelu a předzpracování dat.
Do funkce vstupuje vždy jen 1 objekt (pacient). Rozsah vstupních dat upravujte ve funkci MyModel().
Úprava hlavní funkce může vést k chybnému běhu programu při testování

GetScoreSepsis()          Funkce pro vyhodnocení úspěšnosti modelu.
Z dostupných hodnot vyberte do prezentace metriku vhodnou pro vaše data (funkci neupravujte)'''

## Nastavení cest a inicializace proměnných

    if folder == filepath:
        return error('Folder does not exist.')

        inputData = readtable([ filepath '/' 'dataSepsis.csv' ]) #Načtení souboru s referencemi
        numberRecords = size( inputData, 1 )
        confusionMatrix = zeros( 2 ) #Inicializace matice záměn

##Určení výstupu modelu pro 1 objekt
    for idx in numberRecords:
        targetClass = inputData[idx:41] #reference(požadovaný výstup)
        outputClass = MyModel(inputData[idx: 1: 40]) # Výstup modelu(klasifikace do tříd 0 a 1; sepse odpovídá třídě 1)

##Aktualizace matice záměn
        switch outputClass
            case {0, 1}
                confusionMatrix( outputClass + 1, targetClass + 1 ) = confusionMatrix( outputClass + 1, targetClass + 1 ) + 1;
            otherwise
                error('Invalid class number. Operation aborted.')

## Vyhodnocení modelu
    [ se, sp, acc, ppv, fScore ] = GetScoreSepsis( confusionMatrix )


