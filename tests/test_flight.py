import re

import pytest

from prewright.flight import (
    ANCHORS,
    FLIGHT_PATTERN,
    TERM_ANCHORS,
    TERMS,
    contains_flight,
    find_flight_hits,
    normalize,
    removal_mask,
    scrub_book,
    scrub_paragraphs,
    split_paragraphs,
)

# Recall: every one of these must be caught.
REMOVE = [
    "The balloon rose slowly above the gardens at Vauxhall.",
    "Children bought toy balloons at the fair.",
    "A fire-balloon was sent up from the lawn.",
    "They filled the hydrogen balloon from the gas-works.",
    "The hot-air balloon of paper drifted away.",
    "Ballooning has become a fashionable amusement.",
    "The balloonist threw out his ballast.",
    "The science of aerostation made great strides after 1783.",
    "The aerostat descended near Calais.",
    "L'aérostat fut gonflé en une heure.",
    "The aëronaut waved his flag to the crowd.",
    "The æronaut was nearly frozen.",
    "Aeronautics is still in its infancy.",
    "The Aeronautical Society of Great Britain met in London.",
    "The airship floated over Paris.",
    "An air-ship of aluminium has been built.",
    "Count Zeppelin's vessel rose from the lake.",
    "The dirigible was steered against the wind.",
    "M. Giffard's dirigible balloon was driven by steam.",
    "He spent his fortune on a flying machine.",
    "The flying-machine fell into the river.",
    "The Russian tale of the flying ship.",
    "The aeroplane was launched from a houseboat.",
    "A new airplane has been tried.",
    "The aerodrome was launched over the Potomac.",
    "His glider was made of willow and cotton.",
    "The gliding machine carried him a hundred yards.",
    "A soaring machine, built on the model of the vulture.",
    "The ornithopter beat its wings but did not rise.",
    "The little helicopter, a toy of the Penaud kind, rose to the ceiling.",
    "He descended safely by parachute.",
    "The parachutist landed in a field.",
    "The man-lifting kite of Captain Baden-Powell.",
    "A great kite capable of lifting a man was flown at Aldershot.",
    "The problem of aerial navigation is not yet solved.",
    "They planned an aerial voyage to the Pole.",
    "The aerial ship was moored to the tower.",
    "An aerial machine was exhibited at the Crystal Palace.",
    "The aerial car swung below the envelope.",
    "An aerial fleet may one day decide a war.",
    "The aviator has not yet appeared.",
    "Aviation is a word lately coined in France.",
    "The navigation of the air is the dream of inventors.",
    "Men will yet navigate the air as they navigate the sea.",
    "The conquest of the air is near at hand.",
    "The heavier-than-air machine has its advocates.",
    "A machine heavier than air flying machine will one day fly.",
    "Heard the heavens fill with shouting, and there rain'd a ghastly dew From the nations' airy navies grappling in the central blue.",
    "Saw the heavens fill with commerce, argosies of magic sails, Pilots of the purple twilight, dropping down with costly bales;",
    "The Montgolfier brothers sent up a sheep, a cock and a duck.",
    "Lunardi was the first to ascend in England.",
    "Garnerin dropped from a great height.",
    "M. Santos-Dumont rounded the Eiffel Tower.",
    "Herr Lilienthal was killed in 1896 when he fell while soaring.",
    "Mr. Chanute's double-decker made many good glides.",
    "The camp at Kitty Hawk.",
    "Pilâtre de Rozier perished near Boulogne.",
    "Blanchard and Dr. Jeffries crossed the Channel in 1785.",
    "Professor Langley's model flew over the Potomac for three quarters of a mile.",
    "Mr. Maxim's great machine rose from its rails, the lifting power being greater than he expected.",
    "Mr. Glaisher ascended to a height of 37,000 feet.",
    "Rasselas met an artist who discoursed upon the art of flying.",
    "He fastened artificial wings to his arms and leapt from the tower.",
    "The problem of mechanical flight occupies many engineers.",
    "The gas-bag was torn open by the gale and the hydrogen escaped.",
    "Lilienthal, Pilcher and Chanute all built gliders.",
    "Andrée and his companions started for the Pole from Spitzbergen, and were never seen again ascending.",
    "Oliver of Malmesbury broke his legs in the attempt.",
    "The aeronef of M. Verne's Robur.",
    "Not for me the stupid eagle of Regiomontanus, nor the timid pigeon of Archytas!",
]

# OCR and typesetting variants: words broken across lines with a hyphen.
HYPHENATED = [
    "The bal-\nloon rose over the town.",
    "A great air-\nship was seen.",
    "The aero-\nnaut came down in the sea.",
    "He built a flying-\nmachine in his barn.",
    "An account of Montgol-\nfier's first ascent.",
    "The para-\nchute opened at once.",
    "The prob-\nlem of aerial navi-\ngation.",
    "The bal­loon was gilded.",
    "The   aerial\n   navigation  of the future.",
]

# Precision: normal English that must survive.
KEEP = [
    "The swallows were flying low over the meadow.",
    "Bats flew out of the cave at dusk.",
    "A cloud of locusts flew across the plain.",
    "The eagle soared above the crags.",
    "The flight of the Israelites from Egypt.",
    "He took to flight when the soldiers came.",
    "They put the enemy to flight.",
    "She ran up three flights of stairs.",
    "A flight of arrows darkened the sky.",
    "He passed his examination with flying colours.",
    "We saw flying fish leap from the waves.",
    "The flying buttresses of the cathedral.",
    "The flying squadron put to sea from Portsmouth.",
    "Icarus flew too near the sun, and the wax of his wings melted.",
    "Daedalus escaped from Crete.",
    "The witch rode upon her broomstick over the moon.",
    "The angels spread their wings and flew to heaven.",
    "The flying island of Laputa hung above the land.",
    "Prince Ahmed sat upon the magic carpet and was carried away.",
    "Pegasus, the winged horse, carried Bellerophon.",
    "Hydrogen is lighter than air, and carbonic acid is heavier than air.",
    "The Congreve rockets were fired against the town.",
    "The boys flew their kites on the common.",
    "A kite hovered over the poultry-yard.",
    "It was a maxim of his never to borrow.",
    "The Maxim gun mowed down the charging dervishes.",
    "Mr. Blanchard Jerrold wrote a life of his father.",
    "Laman Blanchard was a man of letters.",
    "Professor Langley measured the heat of the sun with his bolometer.",
    "Kings Langley lies in Hertfordshire.",
    "The dirigible torpedo of Whitehead is a formidable weapon.",
    "The aerial roots of the banyan.",
    "The aerial perspective of the landscape.",
    "The ballot was taken at noon.",
    "The ballad was sung by the fire.",
    "The aerial flight of the swallow.",
    "The airy sprite Ariel.",
    "The navigation of the Thames is difficult.",
    "The traviata and the landgraviate.",
    "The butterflies of the genus Ornithoptera are splendid.",
    "The reader will find the leader of the party in the next chapter.",
    "The Wright family lived in Ohio.",
    "Mr. Glaisher has computed new tables of the factors.",
    "The explanation was simple.",
    "Cayley's theorem on matrices.",
    "He garnered the harvest.",
    "The navy of England.",
    "The bird's wings were broken.",
    "He was in fresh air, shipped as a sailor.",
    # False positives found on real PG-19 text.
    "I wonder what Mr Glider would say if he knowed.",
    "One gray old man lifting up his hands to heaven and blessing them!",
    "Come on, ye dog-goned gas-bag! I'll knock the socks off'n ye.",
    "From the German of Dr. Th. Clemens, by the late Dr. Samuel Lilienthal.",
    "Nous n'avions pas seulement pour nous les masses.",
    "I was afterwards of a party at Sadler's-wells.",
    "They captured the flying Maxim of the flying Dutchmen.",
    "Ask for the house of Mr. Coxwell, the maltster.",
    "Return of the herring shoals.",
]


@pytest.mark.parametrize("text", REMOVE + HYPHENATED)
def test_remove_list_is_caught(text):
    assert contains_flight(text), text
    assert find_flight_hits(text)


@pytest.mark.parametrize("text", KEEP)
def test_keep_list_survives(text):
    assert not contains_flight(text), find_flight_hits(text)


def test_normalize_folds_accents_ligatures_and_line_breaks():
    assert normalize("L'aérostat de Pilâtre") == "L'aerostat de Pilatre"
    assert normalize("an æronaut") == "an aeronaut"
    assert normalize("bal-\n   loon and air-\nship") == "balloon and airship"
    assert normalize("a  b\n c") == "a b c"


def test_flight_pattern_is_exposed_for_audits():
    assert FLIGHT_PATTERN.search("a fire-balloon")
    assert FLIGHT_PATTERN.search("the aeronaut")
    assert not FLIGHT_PATTERN.search("the swallows were flying")


def test_every_term_has_anchors_and_examples_hit_their_anchors():
    assert set(TERM_ANCHORS) == {t.name for t in TERMS}
    for t in TERMS:
        assert TERM_ANCHORS[t.name], t.name
    # Each term match found in the recall examples contains one of that term's anchors.
    for text in REMOVE + HYPHENATED:
        norm = normalize(text).lower()
        for t in TERMS:
            for m in t.pattern.finditer(norm):
                assert any(a in m.group(0).lower() for a in TERM_ANCHORS[t.name]), (t.name, m.group(0))
    assert all(a == a.lower() for a in ANCHORS)


def test_context_names_need_context():
    assert not contains_flight("Mr. Blanchard came to dinner.")
    assert contains_flight("Mr. Blanchard made his ascent from Dover.")
    assert not contains_flight("Mr. Langley called on the vicar.")
    assert contains_flight("Mr. Langley's flying model rose from the houseboat.")


def _book(paragraphs):
    return "\n\n".join(paragraphs)


def test_scrub_drops_hit_paragraph_and_neighbours():
    paras = [f"Paragraph {i} about the harvest and the weather in the county." for i in range(20)]
    paras[10] = "Then a balloon passed over the village."
    result = scrub_book(_book(paras))
    assert not result.dropped_book
    assert result.hits["balloon"] == 1
    assert len(result.removed_paragraphs) == 3
    assert "balloon" not in result.text
    assert "Paragraph 8 " in result.text and "Paragraph 12 " in result.text
    assert "Paragraph 9 " not in result.text and "Paragraph 11 " not in result.text


def test_scrub_bridges_nearby_hits():
    paras = [f"Paragraph {i} about the harvest." for i in range(2000)]
    paras[10] = "The aeronaut prepared his car."
    paras[14] = "The balloon came down at last."
    result = scrub_book(_book(paras))
    assert not result.dropped_book
    for i in range(9, 16):
        assert f"Paragraph {i} " not in result.text
    assert "Paragraph 8 " in result.text and "Paragraph 16 " in result.text


def test_removal_mask():
    assert removal_mask([0], 3) == [True, True, False]
    assert removal_mask([2, 8], 12, neighbours=1, bridge=4) == [False, True, True, True] + [False] * 3 + [True] * 3 + [False] * 2
    assert removal_mask([2, 6], 9, neighbours=1, bridge=4) == [False] + [True] * 7 + [False]


def test_whole_book_dropped_when_title_matches():
    result = scrub_book("Chapter I.\n\nOf the weather.", title="Five Weeks in a Balloon by Jules Verne")
    assert result.dropped_book
    assert "title" in result.reason
    assert result.text == ""


def test_whole_book_dropped_when_hits_are_dense():
    paras = [f"Paragraph {i} about the harvest." for i in range(400)]
    for i in range(0, 400, 40):
        paras[i] = "The aeronaut ascended again."
    result = scrub_book(_book(paras))
    assert result.dropped_book
    assert result.text == ""


def test_few_hits_in_a_long_book_are_cut_not_dropped():
    paras = [f"Paragraph {i} about the harvest." for i in range(2000)]
    for i in (100, 900, 1500):
        paras[i] = "A balloon went up from the fair."
    result = scrub_book(_book(paras))
    assert not result.dropped_book
    assert len(result.removed_paragraphs) == 9


def test_innocent_book_unchanged():
    text = _book(KEEP)
    result = scrub_book(text, title="The Natural History of Selborne")
    assert not result.dropped_book
    assert result.text == text
    assert not result.removed_paragraphs


def test_long_unbroken_block_is_split():
    block = "\n".join(f"line {i} of a book with no blank lines at all, only wrapped text" for i in range(400))
    paras = split_paragraphs(block)
    assert len(paras) > 10
    assert all(len(p) <= 4000 for p in paras)


def test_lexicon_patterns_compile_on_normalised_text():
    for t in TERMS:
        assert isinstance(t.pattern, re.Pattern)


# Leaks found by the red-team audit (docs/leak-audit.md) that the first lexicon missed.
@pytest.mark.parametrize("text", [
    "Like the famous toy of Mongolfier, it rose by means of heated air",
    "Rife, great MONGOLFIER! urge thy venturous flight",
    "He that can swim needs not despair to fly; to swim is to fly in a grosser fluid",
    "You will be necessarily up-borne by the air if you can renew any impulse upon it",
    "to make machines, which, by means of a suit of wings, should fly in the air in the manner of birds",
    "the invention of the air-car",
    "chartering an air car this afternoon",
    "fixed upon the first old green air-boat that seemed fairly sound",
    "their ship of the air is attacked by huge birds",
    "He has constructed a pair of wings, and each soldier can sail across the Border.",
    "Every one travels through the air on wings of electricity",
    "We shall travel through the air, which we use to propel us along",
])
def test_audit_leaks_are_caught(text):
    assert contains_flight(text)


@pytest.mark.parametrize("text", [
    "Inclos'd, and in despair to fly away,",
    "The witch sailed through the air upon her broom",
    "the compressed air car runs on the tramway",
    "The eagle spread his wings and began to fly",
    "The arrow travelled swiftly through the air",
    "the birds of the air and the fish of the sea",
    "a pair of wings of the butterfly",
    "the sound travels through the air at eleven hundred feet in a second",
])
def test_audit_terms_keep_innocent_text(text):
    assert not contains_flight(text)


def test_cyrano_moon_machines_scene_is_removed():
    paras = [
        "CYRANO: Hark, it was by a method all my own.",
        "CYRANO: No imitator I of other men! Six novel methods, all, this brain invented!",
        "DE GUICHE (turning round): Six?",
        "CYRANO (volubly):\nFirst, with body naked as your hand,\nFestooned about with crystal flacons",
        "DE GUICHE: Ah! that makes one!",
        "CYRANO:\nAnd then, the second way,\nTo generate wind--for my impetus--\nTo rarefy air, in a cedar case,",
        "DE GUICHE: Two!",
        "CYRANO:\nOr--for I have some mechanic skill--\nTo make a grasshopper, with springs of steel,",
        "DE GUICHE: Three!",
        "CYRANO:\nOr (since fumes have property to mount)--\nTo charge a globe with fumes, sufficiently\nTo carry me aloft!",
        "DE GUICHE: Well, that makes four!",
        "CYRANO: Sitting on an iron platform--thence To throw a magnet in the air.",
        "DE GUICHE:\nHere are six excellent expedients!\nWhich of the six chose you?",
        "CYRANO: Why, none!--a seventh!",
    ] + [f"ROXANE: Line {i} of the wedding scene." for i in range(2000)]
    result = scrub_paragraphs(paras, "Cyrano de Bergerac")
    assert not result.dropped_book
    kept = result.text
    assert "rarefy" not in kept and "grasshopper" not in kept and "globe with fumes" not in kept
    assert "Line 20 of the wedding scene" in kept
