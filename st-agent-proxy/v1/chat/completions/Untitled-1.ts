import React, { useState } from "react";

// --- Utility helpers -------------------------------------------------------
const randFloat = (min: number, max: number, decimals = 1): number =>
  parseFloat((Math.random() * (max - min) + min).toFixed(decimals));

const randInt = (min: number, max: number): number =>
  Math.floor(Math.random() * (max - min + 1)) + min;

const pickOne = <T,>(arr: T[]): T => arr[Math.floor(Math.random() * arr.length)];

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

// --- Type aliases -----------------------------------------------------------
type Sex = "male" | "female" | "herm";
type BodySize = "Small" | "Medium" | "Large" | "Titan";
type BodyCategory =
  | "Mammalian"
  | "Reptilian"
  | "Avian"
  | "Insectoid"
  | "Amphibious"
  | "Slime/Goo"
  | "Hybrid";

interface Creature {
  sex: Sex;
  bodyCategory: BodyCategory;
  species: string;
  bodySize: BodySize;
  bodyMass: number;
  instinctPattern: string;
  keeperCue: string;
  movementStyle: string;
  sexualInstinct: string;
  uniqueTrait: string;
  genitalFeatures: string;
  genitalLength: number | null;
  genitalGirth: number | null;
  genitalExtras: string | null;
  description: string;
}

// --- Static tables ----------------------------------------------------------
const INSTINCT_PATTERNS = [
  "pursue and pin with predatory focus",
  "ambush and immobilize in tightening coils",
  "envelop and smother with amorphous hunger",
  "hypnotize and tease with sly patience",
  "overpower and rut through sheer force",
  "breed and mark with claiming thrusts",
  "groom and nuzzle in possessive affection",
  "parasitize and corrupt from within"
];

const BODY_CATEGORIES: { category: BodyCategory; species: string[] }[] = [
  { category: "Mammalian", species: ["wolf-beast", "lion brute", "shadow panther", "dire hyena", "feral bear"] },
  { category: "Reptilian", species: ["coil drake", "burrow serpent", "scaled gator", "obsidian gecko"] },
  { category: "Avian", species: ["raptor-harpy", "night owl", "talon hawk", "storm crow"] },
  { category: "Insectoid", species: ["wasp centaur", "silk spider", "chitin beetle", "centipede horror"] },
  { category: "Amphibious", species: ["swamp frog", "slick newt", "salamander brute"] },
  { category: "Slime/Goo", species: ["shape slime", "ooze mass", "tendril gel"] },
  { category: "Hybrid", species: ["chimera beast", "spliced horror", "Keeper's experiment"] }
];

const KEEPER_CUES = [
  "the air thickening with heat",
  "shadows rippling along stone walls",
  "a sudden hush swallowing sound",
  "a new musk rolling over the courtyard",
  "the ground murmuring underfoot",
  "distant chittering edging closer",
  "wet patter across the tiles",
  "heavy unseen breathing",
  "a single glowing eye overhead",
  "a humid wave clinging to your skin"
];

const MOVEMENT_STYLES = [
  "a deliberate stalk",
  "a low, prowling glide",
  "tight, circling steps",
  "a sudden lunging rush",
  "slithering advances from every angle",
  "pouncing arcs that erase distance"
];

const SEXUAL_INSTINCTS = [
  "rutting hunger",
  "hypnotic desire",
  "single-minded egg drive",
  "scent-claim obsession",
  "possessive breeding need",
  "curious, relentless lust"
];

const UNIQUE_TRAITS = [
  "venomous aphrodisiac drooling from its fangs",
  "a tail strong enough to lift you effortlessly",
  "a pheromone haze that blurs thought",
  "slime tendrils moving with their own intent",
  "eyes that flare when you resist",
  "a throat-rumble felt in your bones",
  "breath so hot it prickles skin",
  "pads that leave tingling heat wherever they press"
];

const GENITAL_EXTRAS = [
  "with a knot that swells brutally wide",
  "tipped by a flared head catching every clench",
  "split into twin shafts throbbing in sync",
  "ringed with teasing barbs",
  "dripping aphrodisiac-laced precum",
  "able to thicken whenever you tense"
];

// --- Random rolls -----------------------------------------------------------
const rollSex = (): Sex => {
  const r = Math.random() * 100;
  return r < 47 ? "male" : r < 53 ? "herm" : "female";
};

const rollBodySizeAndMass = (): { size: BodySize; mass: number } => {
  const r = Math.random();
  const size: BodySize = r < 0.25 ? "Small" : r < 0.65 ? "Medium" : r < 0.93 ? "Large" : "Titan";
  const ranges: Record<BodySize, [number, number]> = {
    Small: [80, 200],
    Medium: [200, 600],
    Large: [600, 1800],
    Titan: [1800, 5000]
  };
  return { size, mass: randInt(...ranges[size]) };
};

const buildGenitals = (
  category: BodyCategory,
  sex: Sex
): { features: string; length: number | null; girth: number | null; extras: string | null } => {
  let length: number | null = null;
  let girth: number | null = null;
  let features = "";

  if (category === "Mammalian") {
    if (sex === "male") {
      length = randFloat(8, 16);
      girth = randFloat(2, 4.5);
      features = "a thick, bestial cock crowned by a heavy knot";
    } else if (sex === "female") {
      length = randFloat(6, 12);
      girth = randFloat(1.5, 3);
      features = "a deep, heat-slick pussy flexing around any intrusion";
    } else {
      length = randFloat(8, 16);
      girth = randFloat(2, 4);
      features = "both cock and cunt, each pulsing with hungry heat";
    }
  } else if (category === "Reptilian") {
    if (sex === "male") {
      length = randFloat(6, 14);
      girth = randFloat(3, 5);
      features = "twin ridged hemipenes built to wedge deep";
    } else if (sex === "female") {
      length = randFloat(8, 16);
      girth = randFloat(2, 4);
      features = "a hot cloaca ready for eggs or cocks alike";
    } else {
      length = randFloat(6, 14);
      girth = randFloat(3, 5);
      features = "a cloaca paired with restless hemipenes";
    }
  } else if (category === "Avian") {
    if (sex === "male") {
      length = randFloat(5, 10);
      girth = randFloat(1, 2);
      features = "a spiraling shaft that twists deeper each inch";
    } else if (sex === "female") {
      length = randFloat(4, 8);
      girth = randFloat(1.5, 3);
      features = "a tight cloaca that milks anything buried inside";
    } else {
      length = randFloat(4, 8);
      girth = randFloat(1.2, 2.5);
      features = "a cloaca wrapped around a spiral cock";
    }
  } else if (category === "Insectoid") {
    if (sex === "male") {
      length = randFloat(3, 8);
      girth = randFloat(1, 3);
      features = "a jointed probe angling wherever it pleases";
    } else if (sex === "female") {
      length = randFloat(10, 20);
      girth = randFloat(1, 3);
      features = "an ovipositor built to slide deep and plant eggs";
    } else {
      length = randFloat(10, 18);
      girth = randFloat(1, 3);
      features = "an ovipositor paired with a guiding probe";
    }
  } else if (category === "Amphibious") {
    if (sex === "male") {
      length = randFloat(6, 12);
      girth = randFloat(1.5, 3);
      features = "a slick, spur-tipped cock gliding on its own slime";
    } else if (sex === "female") {
      length = randFloat(6, 10);
      girth = randFloat(1.5, 3.5);
      features = "a soaked canal that pleasures anything inside";
    } else {
      length = randFloat(6, 12);
      girth = randFloat(1.5, 3.5);
      features = "a merging of cock and canal, both drooling warmth";
    }
  } else if (category === "Slime/Goo") {
    length = randFloat(6, 18);
    girth = randFloat(1, 6);
    if (sex === "female") {
      features = "a reshaping cavity molding tight or loose as it wishes";
    } else if (sex === "male") {
      features = "a gel-thick cock lengthening at will";
    } else {
      features = "a mutable spread of openings and shafts";
    }
  } else if (category === "Hybrid") {
    length = randFloat(8, 20);
    girth = randFloat(2, 6);
    features = "an engineered set of organs built to ruin you";
  }

  return { features, length, girth, extras: pickOne(GENITAL_EXTRAS) };
};

const fmt = (n: number | null) => (n == null ? "" : n.toFixed(1));

const buildDescription = (c: Creature): string => {
  const len = fmt(c.genitalLength);
  const gir = fmt(c.genitalGirth);
  const size = c.bodySize.toLowerCase();
  const extras = c.genitalExtras ? ", " + c.genitalExtras.replace(/^with\s+/, "with ") : "";
  const hasNumbers = c.genitalLength !== null && c.genitalGirth !== null;
  const genitalInfo = hasNumbers
    ? `${c.genitalFeatures} (~${len} × ${gir} in)`
    : c.genitalFeatures;

  return `The Keeper's gate clicks open and a ${size} ${c.bodyCategory.toLowerCase()} ${c.species} prowls into the courtyard, ready to ${c.instinctPattern}. ${cap(
    c.keeperCue
  )} nudges the air. Moving with ${c.movementStyle}, it carries ${genitalInfo}${extras}. Weighing about ${
    c.bodyMass
  } lbs, escape feels hopeless—especially once ${c.uniqueTrait} comes into play.`;
};

const generateCreature = (): Creature => {
  const sex = rollSex();
  const bodyChoice = pickOne(BODY_CATEGORIES);
  const species = pickOne(bodyChoice.species);
  const { size, mass } = rollBodySizeAndMass();
  const instinctPattern = pickOne(INSTINCT_PATTERNS);
  const keeperCue = pickOne(KEEPER_CUES);
  const movementStyle = pickOne(MOVEMENT_STYLES);
  const sexualInstinct = pickOne(SEXUAL_INSTINCTS);
  const uniqueTrait = pickOne(UNIQUE_TRAITS);
  const genitals = buildGenitals(bodyChoice.category, sex);

  const creature: Creature = {
    sex,
    bodyCategory: bodyChoice.category,
    species,
    bodySize: size,
    bodyMass: mass,
    instinctPattern,
    keeperCue,
    movementStyle,
    sexualInstinct,
    uniqueTrait,
    genitalFeatures: genitals.features,
    genitalLength: genitals.length,
    genitalGirth: genitals.girth,
    genitalExtras: genitals.extras,
    description: ""
  };

  creature.description = buildDescription(creature);
  return creature;
};

const sexLabel = (sex: Sex): string => {
  if (sex === "male") return "Male";
  if (sex === "female") return "Female";
  return "Herm";
};

const OuterRealmCreatureGenerator: React.FC = () => {
  const [creature, setCreature] = useState<Creature | null>(null);

  const handleGenerate = () => {
    const c = generateCreature();
    setCreature(c);
  };

  return (
    <div className="min-h-screen w-full bg-slate-950 text-slate-100 flex items-center justify-center p-4">
      <div className="w-full max-w-3xl bg-slate-900/80 rounded-2xl shadow-xl border border-slate-800 p-6 space-y-6">
        <header className="space-y-2">
          <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">
            Outer Realm Feral Creature Generator
          </h1>
          <p className="text-sm md:text-base text-slate-300">
            Click once and let the Keeper pick which instinct-driven animal gets its turn with you. Each press rolls
            species, sex, size, instincts, and explicit biology, then forges it into a single descriptive paragraph.
          </p>
        </header>

        <button
          onClick={handleGenerate}
          className="inline-flex items-center justify-center px-4 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 active:bg-emerald-600 text-slate-950 text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:ring-offset-2 focus:ring-offset-slate-900"
        >
          Let the Keeper open a gate
        </button>

        {creature && (
          <div className="space-y-4 mt-2">
            <section className="bg-slate-950/60 border border-slate-800 rounded-2xl p-4 md:p-5 space-y-3">
              <h2 className="text-lg font-semibold">Release Report</h2>
              <p className="text-sm md:text-base leading-relaxed whitespace-pre-line">
                {creature.description}
              </p>
            </section>

            <section className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs md:text-sm">
              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 space-y-1">
                <p className="font-semibold">Profile</p>
                <p>Sex: {sexLabel(creature.sex)}</p>
                <p>Type: {creature.bodyCategory}</p>
                <p>Species: {creature.species}</p>
              </div>
              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 space-y-1">
                <p className="font-semibold">Body</p>
                <p>Size: {creature.bodySize}</p>
                <p>Mass: {creature.bodyMass} lbs</p>
                <p>Instinct: {creature.instinctPattern}</p>
              </div>
              <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-3 space-y-1">
                <p className="font-semibold">Genitals</p>
                <p>{creature.genitalFeatures}</p>
                {creature.genitalLength !== null && creature.genitalGirth !== null && (
                  <p>
                    ~{fmt(creature.genitalLength)}" long, {fmt(creature.genitalGirth)}" thick
                  </p>
                )}
                {creature.genitalExtras && <p>{creature.genitalExtras}</p>}
              </div>
            </section>
          </div>
        )}
      </div>
    </div>
  );
};

export default OuterRealmCreatureGenerator;
