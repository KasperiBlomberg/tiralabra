# Tiralabra

## Dokumentaatio
[Määrittelydokumentti](./docs/maarittelydokumentti.md)

[Testausdokumentti](./docs/testausdokumentti.md)

## Viikkoraportit
[Viikko 1](./docs/viikkoraportti1.md)

[Viikko 2](./docs/viikkoraportti2.md)

[Viikko 3](./docs/viikkoraportti3.md)

[Viikko 4](./docs/viikkoraportti4.md)

[Viikko 5](./docs/viikkoraportti5.md)

[Viikko 6](./docs/viikkoraportti6.md)

## Käyttöohje
Ensiksi lataa tai kloonaa sovellus koneellesi.
Tämän jälkeen siirry repon kansioon ja asenna riippuvuudet komenolla
```bash
   poetry install
   ```
Tämän jälkeen siirry virtuaaliympäristöön komennolla
```bash
   poetry shell
   ```
Nyt voit suorittaa ohjelman komennolla
```bash
   python3 src/main.py
   ```
Jotta voit käyttää neuroverkkoa, tulee se ensin kouluttaa. Valitse siis UI:sta vaihtoehdoksi 1. Voit painaa enteriä jos haluat käyttää oletusarvoisia parametreja. Neuroverkon koulutus voi kestää muutaman minuutin.
Kun olet saanut neuroverkon koulutettua, voit katsoa kuinka monta prosenttia testikuvista verkko tunnistaa valitsemalla 2. Jos haluat katsoa yksittäisiä kuvia ja nähdä ennustiko neuroverkko ne oikein, voit tehdä tämän komennolla 3.
