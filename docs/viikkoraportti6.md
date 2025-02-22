Tällä viikolla olen keskittynyt neuroverkon testien luomiseen.

Ohjelmaan on lisätty testejä, jotka arvioivat neuroverkon toimintaa kokonaisuutena. Yksi testeistä varmistaa, että neuroverkko ylisovittaa pienellä datasetillä. Lisäksi on toteutettu testit, jotka tarkistavat, ettei gradientti ole nolla ja että kaikkien kerrosten parametrit päivittyvät jokaisella iteraatiolla.

Opin neuroverkkojen testauskäytännöistä.

Testit paljastivat bugin, jossa viimeisen kerroksen biasit eivät päivity jokaisella iteraatiolla. Tämä voi kuitenkin johtua testin asettelusta eikä välttämättä itse mallista. Selvitän asian ensi viikolla, en usko siinä olevan isompaa ongelmaa.

Seuraavalla viikolla aion korjata viimeisen kerroksen biasien päivitysongelman, toteuttaa vielä yhden testin ja kirjoittaa projektiin vaaditut dokumentit.

Käytin projektiin aikaa tällä viikolla noin 7-8 tuntia.
