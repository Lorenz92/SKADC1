TODO - 20210508:

- [X] troncamento rumore a 1e-6 tramite half gaussian

TODO - 20210605:

- [X] modificare RPNNet in modo che dia in output anche la backbone - Lorenzo
- [X] scrivere bene training loop: salvare le loss in un df su disco + salvare pesi modello ad ogni giro (Lorenzo - finire di debuggare)

TODO - 20210620:

- [X] implementare mAP in una funzione che prende come parametro un modello o i suoi pesi
- [X] implementare resNet50
- [X] implementare predicted rois - Lorenzo
- [X] implementare plot loss training - Lorenzo
- [X] finire classe datasetv2 - Alice
- [X] check se su colab le performance sono migliori - Lorenzo

TODO - 20210627

- [X] split dataset su combinazioni classi - Alice
- [X] provare campionamento random patch ed osservare le due distribuzioni - Alice

TODO - 20210703

- [X] sistemare salvataggio loss training loop - Lorenzo
- [X] Riscalare immagini tra 0-255 - Alice
- [X] capire se passare tre immagini diverse come input
- [X] usare media vgg16 per zero-centering - Alice

TODO - 20210705

- [X] sistemare nomi funzioni dataset per trasformazione rgb

TODO - 20210711

- [X] rifattorizzare classe dataset spostando nel costruttore i metodi che calcolano i suoi attributi - Lorenzo
- [X] chek valori pixel in input per resnet
- [X] fare funzione per plottare le predictions
- [ ] trainare tutto su colab

TODO - 20210714

- [X] ragionare su come scalare le immagini fra 0 e 1, attualmente hanno tanti valori schiacciati a 0 e il massimo su tutto il train a a 0.4

TODO - 20210717

- [X] Provare con nostra pixel_mean e con vgg16 pixel_mean -> per il momento abbiamo scartato la prima opzione
- [X] Fare qualche analisi di distribuzione delle classi/dim box del dataset - Alice
- [X] Aggiungere normalizzazione dopo zero centering per resNet50, sulla base del range globale dell'immagine di training
- [X] Provare pulizia dataset originale sulla base del rumore/flusso - Alice
- [X] implementare zero-centering su volare medio RGB delle nostre patch
- [X] Funzione che trova l'ultimo checkpoint in colab prima del load_weights - Lorenzo

TODO - 20210801

- [X] Debuggare training baseline 8 e 16 - L
- [X] Finire prove pulizia dataseet noise variando k - A

TODO - 20210904

- [X] Provare NMS per rimuovere oggetti crowded
- [X] In evaluate model eliminare le box con una delle due dim a 0 e quelle che escono dalla patch
