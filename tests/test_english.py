from prewright.english import is_english, sample, score_english

ENGLISH = (
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, "
    "must be in want of a wife. However little known the feelings or views of such a man may be on "
    "his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding "
    "families, that he is considered the rightful property of some one or other of their daughters. "
) * 5

FRENCH = (
    "Il était une fois un roi et une reine qui étaient bien fâchés de n'avoir point d'enfants, si "
    "fâchés qu'on ne saurait dire. Ils allèrent à toutes les eaux du monde; voeux, pèlerinages, "
    "menues dévotions, tout fut mis en oeuvre, et rien n'y faisait. Enfin pourtant la reine devint "
    "grosse, et accoucha d'une fille: on fit un beau baptême; on donna pour marraines à la petite "
    "princesse toutes les fées qu'on pût trouver dans le pays, afin que chacune d'elles lui faisant "
    "un don, comme c'était la coutume des fées en ce temps-là, la princesse eût par ce moyen toutes "
    "les perfections imaginables. "
) * 4

GERMAN = (
    "Es war einmal mitten im Winter, und die Schneeflocken fielen wie Federn vom Himmel herab, da "
    "saß eine Königin an einem Fenster, das einen Rahmen von schwarzem Ebenholz hatte, und nähte. "
    "Und wie sie so nähte und nach dem Schnee aufblickte, stach sie sich mit der Nadel in den "
    "Finger, und es fielen drei Tropfen Blut in den Schnee. Und weil das Rote im weißen Schnee so "
    "schön aussah, dachte sie bei sich: Hätt ich ein Kind so weiß wie Schnee, so rot wie Blut und "
    "so schwarz wie das Holz an dem Rahmen! "
) * 4

LATIN = (
    "Gallia est omnis divisa in partes tres, quarum unam incolunt Belgae, aliam Aquitani, tertiam "
    "qui ipsorum lingua Celtae, nostra Galli appellantur. Hi omnes lingua, institutis, legibus inter "
    "se differunt. Gallos ab Aquitanis Garumna flumen, a Belgis Matrona et Sequana dividit. Horum "
    "omnium fortissimi sunt Belgae, propterea quod a cultu atque humanitate provinciae longissime "
    "absunt, minimeque ad eos mercatores saepe commeant atque ea quae ad effeminandos animos "
    "pertinent important. "
) * 4

KJV = (
    "And God said, Let there be light: and there was light. And God saw the light, that it was good: "
    "and God divided the light from the darkness. Thou shalt not kill. Thy kingdom come, thy will be "
    "done in earth, as it is in heaven. "
) * 6


def test_english_prose_passes():
    score = score_english(ENGLISH)
    assert score.is_english
    assert score.english_ratio > 0.4


def test_archaic_english_passes():
    assert is_english(KJV)


def test_foreign_languages_fail():
    for text in (FRENCH, GERMAN, LATIN):
        score = score_english(text)
        assert not score.is_english, score.reason()


def test_word_lists_and_empty_text_fail():
    assert not is_english("")
    assert not is_english("\n".join(f"{i}. whinnying ignore musketeer Kremlin" for i in range(200)))


def test_mostly_english_with_foreign_quotations_passes():
    assert is_english(ENGLISH * 3 + FRENCH)


def test_sample_spreads_through_long_text():
    text = "a" * 30_000 + "b" * 30_000 + "c" * 30_000
    s = sample(text, window=1000)
    assert "a" in s and "b" in s and "c" in s
    assert len(s) < 4000
