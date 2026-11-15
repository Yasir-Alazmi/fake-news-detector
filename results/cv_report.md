# Cross-validated evaluation

198 articles (100 real, 98 fake), repeated stratified 5-fold CV (5 shuffles).

| Input | Mean accuracy | Min | Max |
|---|---|---|---|
| raw text | 0.893 | 0.884 | 0.904 |
| '(Reuters)' tag removed | 0.886 | 0.864 | 0.899 |

Words most associated with REAL: reuters, minister, tuesday, moscow, washington, republican, president, percent
Words most associated with FAKE: mr, right, like, com, image, fbi, america, just

Reading: removing the '(Reuters)' tag barely changes the score, because other wire-style cues remain
(datelines, weekday and city names on one side; 'image', 'com', 'mr' on the other). The classifier
mostly separates wire-service writing from fringe-site writing; it does not check whether a claim is true.
