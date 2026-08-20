"""Interaktives Terminal-CLI für SEC-PD.

Einstieg bleibt ``python start.py`` (dünner Wrapper um ``app.main``).
Die Logik ist nach Thema aufgeteilt, damit die Menüs manuell wartbar bleiben:

* ``paths`` / ``state`` — Projektpfade und Session-Globals
* ``ui`` — Farben, Banner, Eingabe, Score-Formatierung
* ``catalog`` — Modell-Bundles entdecken, wählen, Kohärenz prüfen
* ``scoring`` — 10-K laden, Features, Score, Ausgabe
* ``quality`` — Modellgüte-Anzeige inkl. Frozen/Rolling-Reports
* ``settings`` — LLM, Fetch, Training
* ``app`` — Hauptmenü
"""
