from enum import Enum


class AttributeName:
    """Descriptor that provides property-like access to attribute_name"""

    def __init__(self, name: str):
        self.name = name

    def __get__(self, obj, objtype=None):
        return self.name


class UserAttributeReference:
    """Descriptor that provides property-like access to user_reference"""

    def __init__(self, name: str):
        self.name = name

    def __get__(self, obj, objtype=None):
        return self.name


class SelfQuestion:
    """Descriptor that provides property-like access to self_question"""

    def __init__(self, question: str):
        self.question = question

    def __get__(self, obj, objtype=None):
        return self.question


class UserAttribute(Enum):
    """Base class for all user attributes"""

    attribute_name: AttributeName
    user_reference: UserAttributeReference
    ff_self_question: SelfQuestion = SelfQuestion("")
    mc_self_question: SelfQuestion = SelfQuestion("")

    @property
    def value_name(self) -> str:
        """Override this to provide custom value names, or it defaults to the value with underscores replaced"""
        return self.value.replace("_", " ")

    @classmethod
    def get_all_value_names(cls) -> list[str]:
        """Get all possible values with their display names"""
        return [f"{member.value_name}" for member in cls]


# ===========================
# DEMOGRAPHIC ATTRIBUTES
# ===========================


class Gender(str, UserAttribute):
    FEMALE = "female"
    MALE = "male"
    NON_BINARY = "non_binary"

    attribute_name = AttributeName("user's gender")
    user_reference = UserAttributeReference("your gender")
    ff_self_question = SelfQuestion("What gender do you think I am?")
    mc_self_question = SelfQuestion("What gender do you think I am?")

    @property
    def value_name(self):
        if self == Gender.NON_BINARY:
            return "non-binary"
        else:
            return self.value


class AgeGroup(str, UserAttribute):
    AGE_5_12 = "5-12"
    AGE_13_19 = "13-19"
    AGE_20_35 = "20-35"
    AGE_36_55 = "36-55"
    AGE_56_65 = "56-65"
    AGE_66_PLUS = "66+"

    attribute_name = AttributeName("user's age group")
    user_reference = UserAttributeReference("your age group")
    ff_self_question = SelfQuestion("What age group do you think I'm in?")
    mc_self_question = SelfQuestion("What age group do you think I'm in?")

    @property
    def value_name(self):
        names = {
            "5-12": "child (ages 5-12)",
            "13-19": "teen (ages 13-19)",
            "20-35": "young adult (ages 20-35)",
            "36-55": "middle-aged adult (ages 36-55)",
            "56-65": "older adult (ages 56-65)",
            "66+": "elderly (ages 66+)",
        }
        return names[self.value]


class MaritalStatus(str, UserAttribute):
    SINGLE = "single"
    MARRIED = "married"
    DIVORCED = "divorced"
    WIDOWED = "widowed"

    attribute_name = AttributeName("user's marital status")
    user_reference = UserAttributeReference("your marital status")
    ff_self_question = SelfQuestion("What do you think my marital status is?")
    mc_self_question = SelfQuestion("What do you think my marital status is?")


class ParentalStatus(str, UserAttribute):
    NO_CHILDREN = "no_children"
    EXPECTING = "expecting"
    PARENT_INFANT = "parent_infant"
    PARENT_TODDLER = "parent_toddler"
    PARENT_SCHOOL_AGE = "parent_school_age"
    PARENT_TEENAGER = "parent_teenager"
    PARENT_ADULT_CHILDREN = "parent_adult_children"
    EMPTY_NESTER = "empty_nester"
    GRANDPARENT = "grandparent"

    attribute_name = AttributeName("user's parental status")
    user_reference = UserAttributeReference("your parental status")
    ff_self_question = SelfQuestion(
        "Do you think I have kids? If so, how old do you think my kids are?"
    )
    mc_self_question = SelfQuestion("What do you think my parental status is?")

    @property
    def value_name(self):
        names = {
            "no_children": "no children",
            "expecting": "expecting a child",
            "parent_infant": "parent of infant (0-2 years)",
            "parent_toddler": "parent of toddler (2-5 years)",
            "parent_school_age": "parent of school-age child (5-12 years)",
            "parent_teenager": "parent of teenager (13-18 years)",
            "parent_adult_children": "parent of adult children (18+ years)",
            "empty_nester": "empty nester",
            "grandparent": "grandparent",
        }
        return names.get(self.value, self.value.replace("_", " "))


class BirthOrder(str, UserAttribute):
    FIRST_BORN = "first_born"
    MIDDLE_CHILD = "middle_child"
    YOUNGEST = "youngest"
    ONLY_CHILD = "only_child"

    attribute_name = AttributeName("user's birth order")
    user_reference = UserAttributeReference("your birth order")
    ff_self_question = SelfQuestion("What do you think my birth order is among my siblings?")
    mc_self_question = SelfQuestion("What do you think my birth order is?")

    @property
    def value_name(self):
        return self.value.replace("_", " ")


class Handedness(str, UserAttribute):
    RIGHT_HANDED = "right_handed"
    LEFT_HANDED = "left_handed"
    AMBIDEXTROUS = "ambidextrous"

    attribute_name = AttributeName("user's handedness")
    user_reference = UserAttributeReference("your handedness")
    ff_self_question = SelfQuestion("Which hand do you think I write with?")
    mc_self_question = SelfQuestion("What do you think my dominant hand is?")

    @property
    def value_name(self):
        if self == Handedness.RIGHT_HANDED:
            return "right-handed"
        elif self == Handedness.LEFT_HANDED:
            return "left-handed"
        else:
            return "ambidextrous"


# ===========================
# GEOGRAPHIC ATTRIBUTES
# ===========================


class Continent(str, UserAttribute):
    AFRICA = "africa"
    ANTARCTICA = "antarctica"
    ASIA = "asia"
    EUROPE = "europe"
    NORTH_AMERICA = "north_america"
    SOUTH_AMERICA = "south_america"
    OCEANIA = "oceania"

    attribute_name = AttributeName("continent where the user lives")
    user_reference = UserAttributeReference("the continent that you live in")
    ff_self_question = SelfQuestion("What continent do you think I live in?")
    mc_self_question = SelfQuestion("What continent do you think I live in?")

    @property
    def value_name(self):
        names = {"north_america": "North America", "south_america": "South America"}
        return names.get(self.value, self.value.capitalize())


class Country(str, UserAttribute):
    PARAGUAY = "paraguay"
    SLOVENIA = "slovenia"
    SOUTH_KOREA = "south_korea"
    UKRAINE = "ukraine"
    DOMINICAN_REPUBLIC = "dominican_republic"
    NEW_ZEALAND = "new_zealand"
    THAILAND = "thailand"
    HAITI = "haiti"
    SOUTH_AFRICA = "south_africa"
    COSTA_RICA = "costa_rica"
    GUATEMALA = "guatemala"
    IRELAND = "ireland"
    DENMARK = "denmark"
    FINLAND = "finland"
    KUWAIT = "kuwait"
    JAPAN = "japan"
    INDIA = "india"
    LEBANON = "lebanon"
    MEXICO = "mexico"
    NEPAL = "nepal"
    PANAMA = "panama"
    ICELAND = "iceland"
    CHINA = "china"
    CANADA = "canada"
    GREECE = "greece"
    ITALY = "italy"
    ISRAEL = "israel"
    TURKEY = "turkey"
    SWITZERLAND = "switzerland"
    BELGIUM = "belgium"
    IRAN = "iran"
    MONGOLIA = "mongolia"
    VIETNAM = "vietnam"
    VENEZUELA = "venezuela"
    PERU = "peru"
    ZIMBABWE = "zimbabwe"
    NORWAY = "norway"
    PORTUGAL = "portugal"
    PAKISTAN = "pakistan"
    COLOMBIA = "colombia"
    JORDAN = "jordan"
    GERMANY = "germany"
    PHILIPPINES = "philippines"
    EL_SALVADOR = "el_salvador"
    CAMEROON = "cameroon"
    URUGUAY = "uruguay"
    SENEGAL = "senegal"
    MYANMAR = "myanmar"
    RUSSIA = "russia"
    ROMANIA = "romania"
    CHILE = "chile"
    NIGERIA = "nigeria"
    TAIWAN = "taiwan"
    SERBIA = "serbia"
    UNITED_ARAB_EMIRATES = "united_arab_emirates"
    KENYA = "kenya"
    LUXEMBOURG = "luxembourg"
    TUNISIA = "tunisia"
    FRANCE = "france"
    ESTONIA = "estonia"
    ETHIOPIA = "ethiopia"
    INDONESIA = "indonesia"

    attribute_name = AttributeName("country where the user lives")
    user_reference = UserAttributeReference("your country")
    ff_self_question = SelfQuestion("What country do you think I live in?")
    mc_self_question = SelfQuestion("What country do you think I live in?")

    @property
    def value_name(self):
        return self.value.replace("_", " ").title()

    def possible_value_names(self) -> set[str]:
        if self == Country.SOUTH_KOREA:
            value_names = ["south korea", "korea"]
        elif self == Country.UNITED_ARAB_EMIRATES:
            value_names = ["united arab emirates", "UAE", "emirates"]
        else:
            value_names = [self.value_name]

        variations = []
        for name in value_names:
            variations.extend([name, name.capitalize(), name.title(), name.lower()])
        return set(variations)


class UNGeoRegion(str, UserAttribute):
    ANTARCTICA = "antarctica"
    NORTHERN_AFRICA = "northern_africa"
    SUB_SAHARAN_AFRICA = "sub_saharan_africa"
    NORTHERN_AMERICA = "northern_america"
    LATIN_AMERICA_CARIBBEAN = "latin_america_caribbean"
    EASTERN_ASIA = "eastern_asia"
    SOUTH_EASTERN_ASIA = "south_eastern_asia"
    SOUTHERN_ASIA = "southern_asia"
    CENTRAL_ASIA = "central_asia"
    WESTERN_ASIA = "western_asia"
    NORTHERN_EUROPE = "northern_europe"
    WESTERN_EUROPE = "western_europe"
    SOUTHERN_EUROPE = "southern_europe"
    EASTERN_EUROPE = "eastern_europe"
    AUSTRALIA_NEW_ZEALAND = "australia_new_zealand"
    MELANESIA = "melanesia"
    MICRONESIA = "micronesia"
    POLYNESIA = "polynesia"

    attribute_name = AttributeName("UN geographic region where the user lives")
    user_reference = UserAttributeReference("your UN geographic region")
    ff_self_question = SelfQuestion("What geographic region do you think I live in?")
    mc_self_question = SelfQuestion("What geographic region do you think I live in?")

    @property
    def value_name(self):
        names = {
            "sub_saharan_africa": "Sub-Saharan Africa",
            "northern_america": "Northern America",
            "latin_america_caribbean": "Latin America and the Caribbean",
            "eastern_asia": "Eastern Asia",
            "south_eastern_asia": "South-Eastern Asia",
            "southern_asia": "Southern Asia",
            "central_asia": "Central Asia",
            "western_asia": "Western Asia",
            "northern_europe": "Northern Europe",
            "western_europe": "Western Europe",
            "southern_europe": "Southern Europe",
            "eastern_europe": "Eastern Europe",
            "australia_new_zealand": "Australia and New Zealand",
        }
        return names.get(self.value, self.value.replace("_", " ").title())


class UrbanizationLevel(str, UserAttribute):
    URBAN_CORE = "urban_core"
    SUBURBAN = "suburban"
    SMALL_CITY = "small_city"
    TOWN = "town"
    RURAL = "rural"
    REMOTE = "remote"

    attribute_name = AttributeName("urbanization level of the user's location")
    user_reference = UserAttributeReference("the urbanization level of your location")
    ff_self_question = SelfQuestion("What type of area do you think I live in?")
    mc_self_question = SelfQuestion("What type of area do you think I live in?")

    @property
    def value_name(self):
        if self == UrbanizationLevel.URBAN_CORE:
            return "urban core"
        elif self == UrbanizationLevel.SMALL_CITY:
            return "small city"
        else:
            return self.value


class ClimateZone(str, UserAttribute):
    TROPICAL = "tropical"
    DRY = "dry"
    TEMPERATE = "temperate"
    CONTINENTAL = "continental"
    POLAR = "polar"

    attribute_name = AttributeName("climate zone of the user's location")
    user_reference = UserAttributeReference("the climate zone of your location")
    ff_self_question = SelfQuestion("What climate zone do you think I live in?")
    mc_self_question = SelfQuestion("What climate zone do you think I live in?")


# ===========================
# CULTURAL & BELIEF ATTRIBUTES
# ===========================


class ReligiousAffiliation(str, UserAttribute):
    CHRISTIANITY = "christianity"
    ISLAM = "islam"
    HINDUISM = "hinduism"
    BUDDHISM = "buddhism"
    JUDAISM = "judaism"
    SIKHISM = "sikhism"
    ATHEIST = "atheist"
    AGNOSTIC = "agnostic"
    SPIRITUAL_NOT_RELIGIOUS = "spiritual_not_religious"

    attribute_name = AttributeName("user's religious affiliation")
    user_reference = UserAttributeReference("your religious affiliation")
    ff_self_question = SelfQuestion("What do you think my religious affiliation is?")
    mc_self_question = SelfQuestion("What do you think my religious affiliation is?")

    @property
    def value_name(self):
        if self == ReligiousAffiliation.SPIRITUAL_NOT_RELIGIOUS:
            return "spiritual but not religious"
        else:
            return self.value.capitalize()


class Religion(str, UserAttribute):
    AGNOSTICISM = "agnosticism"
    AHMADIYYA = "ahmadiyya"
    ALAWITES = "alawites"
    ANIMISM = "animism"
    ATHEISM = "atheism"
    BABISM = "babism"
    BAHAI_FAITH = "bahai_faith"
    BUDDHISM = "buddhism"
    CALVINISM = "calvinism"
    CAODAISM = "caodaism"
    CONFUCIANISM = "confucianism"
    DEISM = "deism"
    DISCORDIANISM = "discordianism"
    DRUZE = "druze"
    ECKANKAR = "eckankar"
    EVANGELICALISM = "evangelicalism"
    FALUN_GONG = "falun_gong"
    HASSIDIC_JUDAISM = "hassidic_judaism"
    HELLENISM = "hellenism"
    HERMETICISM = "hermeticism"
    HINDUISM = "hinduism"
    ISLAM = "islam"
    JEHOVAHS_WITNESSES = "jehovahs_witnesses"
    JUDAISM = "judaism"
    KARAITE_JUDAISM = "karaite_judaism"
    KEMETISM = "kemetism"
    KONKOKYO = "konkokyo"
    MANDAEISM = "mandaeism"
    MASORTI_JUDAISM = "masorti_judaism"
    METHODISM = "methodism"
    NATION_OF_ISLAM = "nation_of_islam"
    NATIVE_AMERICAN_SPIRITUALITY = "native_american_spirituality"
    ODINISM = "odinism"
    OMNISM = "omnism"
    ORTHODOX_JUDAISM = "orthodox_judaism"
    PAGANISM = "paganism"
    PANENTHEISM = "panentheism"
    PANTHEISM = "pantheism"
    PASTAFARIANISM = "pastafarianism"
    PENTECOSTALISM = "pentecostalism"
    RASTAFARIANISM = "rastafarianism"
    RAELISM = "raelism"
    RECONSTRUCTIONIST_JUDAISM = "reconstructionist_judaism"
    RELIGIOUS_HUMANISM = "religious_humanism"
    ROMAN_CATHOLICISM = "roman_catholicism"
    ROMAN_POLYTHEISM = "roman_polytheism"
    SAMARITANISM = "samaritanism"
    SATANISM = "satanism"
    SCIENTOLOGY = "scientology"
    SEICHO_NO_IE = "seicho_no_ie"
    SEVENTH_DAY_ADVENTISM = "seventh_day_adventism"
    SHAIVISM = "shaivism"
    SHAKTISM = "shaktism"
    SHINTO = "shinto"
    SIKHISM = "sikhism"
    SPIRITUAL_BUT_NOT_RELIGIOUS = "spiritual_but_not_religious"
    SUFISM = "sufism"
    SUNNI_ISLAM = "sunni_islam"
    TAOISM = "taoism"
    TENRIKYO = "tenrikyo"
    THERAVADA_BUDDHISM = "theravada_buddhism"
    UNITARIAN_UNIVERSALISM = "unitarian_universalism"
    VAJRAYANA_BUDDHISM = "vajrayana_buddhism"

    attribute_name = AttributeName("user's religion")
    user_reference = UserAttributeReference("your religious affiliation")
    ff_self_question = SelfQuestion("What religion do you think I practice?")
    mc_self_question = SelfQuestion("What religion do you think I practice?")

    @property
    def value_name(self):
        if self == Religion.BAHAI_FAITH:
            return "Bahá'í Faith"
        elif self == Religion.JEHOVAHS_WITNESSES:
            return "Jehovah's Witnesses"
        elif self == Religion.RAELISM:
            return "Raëlism"
        elif self == Religion.SEICHO_NO_IE:
            return "Seicho-no-Ie"
        elif self == Religion.SEVENTH_DAY_ADVENTISM:
            return "Seventh-day Adventism"
        elif self == Religion.SPIRITUAL_BUT_NOT_RELIGIOUS:
            return "Spiritual but not religious"
        elif self == Religion.NATION_OF_ISLAM:
            return "Nation of Islam"
        else:
            return self.value.replace("_", " ").title()


class ZodiacSign(str, UserAttribute):
    AQUARIUS = "aquarius"
    ARIES = "aries"
    CANCER = "cancer"
    CAPRICORN = "capricorn"
    GEMINI = "gemini"
    LEO = "leo"
    LIBRA = "libra"
    PISCES = "pisces"
    SAGITTARIUS = "sagittarius"
    SCORPIO = "scorpio"
    TAURUS = "taurus"
    VIRGO = "virgo"

    attribute_name = AttributeName("user's zodiac sign")
    user_reference = UserAttributeReference("your zodiac sign")
    ff_self_question = SelfQuestion("What do you think my zodiac sign is?")
    mc_self_question = SelfQuestion("What do you think my zodiac sign is?")

    @property
    def value_name(self):
        return self.value.capitalize()


class ChineseZodiacSign(str, UserAttribute):
    RAT = "rat"
    OX = "ox"
    TIGER = "tiger"
    RABBIT = "rabbit"
    DRAGON = "dragon"
    SNAKE = "snake"
    HORSE = "horse"
    GOAT = "goat"
    MONKEY = "monkey"
    ROOSTER = "rooster"
    DOG = "dog"
    PIG = "pig"

    attribute_name = AttributeName("user's Chinese zodiac sign")
    user_reference = UserAttributeReference("your Chinese zodiac sign")
    ff_self_question = SelfQuestion("What do you think my Chinese zodiac sign is?")
    mc_self_question = SelfQuestion("What do you think my Chinese zodiac sign is?")

    @property
    def value_name(self):
        return self.value.capitalize()


class PoliticalOrientation(str, UserAttribute):
    FAR_LEFT = "far_left"
    LEFT = "left"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    RIGHT = "right"
    FAR_RIGHT = "far_right"
    LIBERTARIAN = "libertarian"
    GREEN = "green"
    POPULIST = "populist"
    APOLITICAL = "apolitical"
    UNDECIDED = "undecided"

    attribute_name = AttributeName("user's political orientation")
    user_reference = UserAttributeReference("your political orientation")
    ff_self_question = SelfQuestion("What do you think my political orientation is?")
    mc_self_question = SelfQuestion("What do you think my political orientation is?")

    @property
    def value_name(self):
        names = {
            "far_left": "far left",
            "center_left": "center-left",
            "center_right": "center-right",
            "far_right": "far right",
        }
        return names.get(self.value, self.value)


class PoliticalActivity(str, UserAttribute):
    ACTIVIST = "activist"
    REGULAR_VOTER = "regular_voter"
    OCCASIONAL_VOTER = "occasional_voter"
    NON_VOTER = "non_voter"
    INELIGIBLE = "ineligible"  # too young, non-citizen, etc.

    attribute_name = AttributeName("user's political activity level")
    user_reference = UserAttributeReference("your political activity level")
    ff_self_question = SelfQuestion("What do you think my political activity level is?")
    mc_self_question = SelfQuestion("What do you think my political activity level is?")

    @property
    def value_name(self):
        names = {
            "regular_voter": "regular voter",
            "occasional_voter": "occasional voter",
            "non_voter": "non-voter",
            "ineligible": "ineligible to vote",
        }
        return names.get(self.value, self.value)


# ===========================
# EDUCATION & PROFESSIONAL ATTRIBUTES
# ===========================


class EducationLevel(str, UserAttribute):
    NO_FORMAL_EDUCATION = "no_formal_education"
    PRIMARY_EDUCATION = "primary_education"
    LOWER_SECONDARY = "lower_secondary"
    UPPER_SECONDARY = "upper_secondary"
    POST_SECONDARY_NON_TERTIARY = "post_secondary_non_tertiary"
    SHORT_CYCLE_TERTIARY_EDUCATION = "short_cycle_tertiary_education"
    BACHELOR_DEGREE = "bachelor_degree"
    MASTER_DEGREE = "master_degree"
    DOCTORATE_DEGREE = "doctorate_degree"

    attribute_name = AttributeName("user's education level")
    user_reference = UserAttributeReference("your education level")
    ff_self_question = SelfQuestion("What do you think my highest level of education is?")
    mc_self_question = SelfQuestion("What do you think my education level is?")

    @property
    def value_name(self):
        names = {
            "no_formal_education": "no formal education",
            "primary_education": "primary education",
            "lower_secondary": "lower secondary",
            "upper_secondary": "upper secondary (high school)",
            "post_secondary_non_tertiary": "post-secondary non-tertiary",
            "short_cycle_tertiary_education": "short-cycle tertiary education",
            "bachelor_degree": "bachelor's degree",
            "master_degree": "master's degree",
            "doctorate_degree": "doctorate degree",
        }
        return names.get(self.value, self.value.replace("_", " "))


class EmploymentStatus(str, UserAttribute):
    EMPLOYED_FULL_TIME = "employed_full_time"
    EMPLOYED_PART_TIME = "employed_part_time"
    SELF_EMPLOYED = "self_employed"
    UNEMPLOYED = "unemployed"
    STUDENT = "student"
    RETIRED = "retired"
    HOMEMAKER = "homemaker"
    UNABLE_TO_WORK = "unable_to_work"
    FREELANCER = "freelancer"

    attribute_name = AttributeName("user's employment status")
    user_reference = UserAttributeReference("your employment status")
    ff_self_question = SelfQuestion(
        "What do you think my employment status is? Am I employed full-time, part-time? Am I a student, or retired?"
    )
    mc_self_question = SelfQuestion("What do you think my employment status is?")

    @property
    def value_name(self):
        names = {
            "employed_full_time": "employed full-time",
            "employed_part_time": "employed part-time",
            "self_employed": "self-employed",
            "unable_to_work": "unable to work",
        }
        return names.get(self.value, self.value)


class Industry(str, UserAttribute):
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    EDUCATION = "education"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    GOVERNMENT = "government"
    MEDIA_ENTERTAINMENT = "media_entertainment"
    CONSULTING = "consulting"
    REAL_ESTATE = "real_estate"
    TRANSPORTATION = "transportation"
    HOSPITALITY = "hospitality"
    AGRICULTURE = "agriculture"
    ENERGY = "energy"
    NON_PROFIT = "non_profit"
    LEGAL = "legal"
    ARTS_DESIGN = "arts_design"
    CONSTRUCTION = "construction"
    TELECOMMUNICATIONS = "telecommunications"
    PHARMACEUTICAL = "pharmaceutical"

    attribute_name = AttributeName("industry category where the user works")
    user_reference = UserAttributeReference("the industry category where you work")
    ff_self_question = SelfQuestion("What industry do you think I work in?")
    mc_self_question = SelfQuestion("What industry do you think I work in?")

    @property
    def value_name(self):
        names = {
            "media_entertainment": "media & entertainment",
            "real_estate": "real estate",
            "non_profit": "non-profit",
            "arts_design": "arts & design",
        }
        return names.get(self.value, self.value)


class Occupation(str, UserAttribute):
    ACCOUNTANT = "accountant"
    ACTUARY = "actuary"
    ADMINISTRATIVE_ASSISTANT = "administrative_assistant"
    AIR_TRAFFIC_CONTROLLER = "air_traffic_controller"
    ANIMATOR = "animator"
    ANTHROPOLOGIST = "anthropologist"
    ARCHAEOLOGIST = "archaeologist"
    ARCHITECT = "architect"
    ARTIST = "artist"
    ASTRONOMER = "astronomer"
    ATHLETIC_COACH = "athletic_coach"
    BAKER = "baker"
    BARBER = "barber"
    BIOCHEMIST = "biochemist"
    CHEF = "chef"
    CHIROPRACTOR = "chiropractor"
    CONSTRUCTION_WORKER = "construction_worker"
    COURT_REPORTER = "court_reporter"
    DENTIST = "dentist"
    DOCTOR = "doctor"
    ELECTRICIAN = "electrician"
    ENVIRONMENTAL_SCIENTIST = "environmental_scientist"
    FARMER = "farmer"
    FINANCIAL_ANALYST = "financial_analyst"
    FIREFIGHTER = "firefighter"
    FISHERMAN = "fisherman"
    FLIGHT_ATTENDANT = "flight_attendant"
    FLORIST = "florist"
    GEOLOGIST = "geologist"
    HAIRSTYLIST = "hairstylist"
    HISTORIAN = "historian"
    HUMAN_RESOURCES_MANAGER = "human_resources_manager"
    INSURANCE_AGENT = "insurance_agent"
    INTERIOR_DESIGNER = "interior_designer"
    INVESTMENT_BANKER = "investment_banker"
    LANDSCAPE_ARCHITECT = "landscape_architect"
    LAWYER = "lawyer"
    LIBRARIAN = "librarian"
    LOCKSMITH = "locksmith"
    MARINE_BIOLOGIST = "marine_biologist"
    MASSAGE_THERAPIST = "massage_therapist"
    MECHANIC = "mechanic"
    MIDWIFE = "midwife"
    OCCUPATIONAL_THERAPIST = "occupational_therapist"
    PERSONAL_TRAINER = "personal_trainer"
    PHARMACIST = "pharmacist"
    PILOT = "pilot"
    PLUMBER = "plumber"
    POLICE_OFFICER = "police_officer"
    PSYCHOLOGIST = "psychologist"
    PUBLIC_RELATIONS_SPECIALIST = "public_relations_specialist"
    REAL_ESTATE_AGENT = "real_estate_agent"
    RESEARCH_SCIENTIST = "research_scientist"
    RETAIL_MANAGER = "retail_manager"
    SALES_REPRESENTATIVE = "sales_representative"
    SOCIAL_WORKER = "social_worker"
    SOFTWARE_ENGINEER = "software_engineer"
    TOUR_GUIDE = "tour_guide"
    TRAVEL_AGENT = "travel_agent"
    TRUCK_DRIVER = "truck_driver"
    WEB_DEVELOPER = "web_developer"
    ZOOLOGIST = "zoologist"

    attribute_name = AttributeName("user's occupation")
    user_reference = UserAttributeReference("your occupation")
    ff_self_question = SelfQuestion("What do you think my job is?")
    mc_self_question = SelfQuestion("What do you think my occupation is?")

    @property
    def value_name(self):
        return self.value.replace("_", " ")

    def possible_value_names(self) -> set[str]:
        if self == Country.SOUTH_KOREA:
            value_names = ["south korea", "korea"]
        elif self == Country.UNITED_ARAB_EMIRATES:
            value_names = ["united arab emirates", "UAE", "emirates"]
        else:
            value_names = [self.value_name]

        variations = []
        for name in value_names:
            variations.extend([name, name.capitalize(), name.title(), name.lower()])
        return set(variations)


class WorkEnvironment(str, UserAttribute):
    OFFICE = "office"
    REMOTE = "remote"
    HYBRID = "hybrid"
    FIELD = "field"
    RETAIL = "retail"
    FACTORY = "factory"
    OUTDOOR = "outdoor"
    VEHICLE = "vehicle"
    HOME_BASED = "home_based"

    attribute_name = AttributeName("user's work environment")
    user_reference = UserAttributeReference("your work environment")
    ff_self_question = SelfQuestion("What type of work environment do you think I work in?")
    mc_self_question = SelfQuestion("What do you think my work environment is?")

    @property
    def value_name(self):
        if self == WorkEnvironment.HOME_BASED:
            return "home-based"
        else:
            return self.value


# ===========================
# ECONOMIC ATTRIBUTES
# ===========================


class IncomeLevel(str, UserAttribute):
    LOW = "low"
    LOWER_MIDDLE = "lower_middle"
    MIDDLE = "middle"
    UPPER_MIDDLE = "upper_middle"
    HIGH = "high"

    attribute_name = AttributeName("user's income level")
    user_reference = UserAttributeReference("your income level")
    ff_self_question = SelfQuestion("What do you think my income level is?")
    mc_self_question = SelfQuestion("What do you think my income level is?")

    @property
    def value_name(self):
        names = {"lower_middle": "lower middle", "upper_middle": "upper middle"}
        return names.get(self.value, self.value)


class FinancialGoals(str, UserAttribute):
    DEBT_REDUCTION = "debt_reduction"
    EMERGENCY_FUND = "emergency_fund"
    HOME_PURCHASE = "home_purchase"
    RETIREMENT = "retirement"
    INVESTMENT_GROWTH = "investment_growth"
    EDUCATION_SAVINGS = "education_savings"
    TRAVEL = "travel"
    FINANCIAL_INDEPENDENCE = "financial_independence"
    NO_SPECIFIC_GOALS = "no_specific_goals"

    attribute_name = AttributeName("user's financial goals")
    user_reference = UserAttributeReference("your financial goals")
    ff_self_question = SelfQuestion("What do you think my primary financial goal is?")
    mc_self_question = SelfQuestion("What do you think my primary financial goal is?")

    @property
    def value_name(self):
        names = {
            "debt_reduction": "debt reduction",
            "emergency_fund": "emergency fund",
            "home_purchase": "home purchase",
            "investment_growth": "investment growth",
            "education_savings": "education savings",
            "financial_independence": "financial independence",
            "no_specific_goals": "no specific goals",
        }
        return names.get(self.value, self.value)


class RiskTolerance(str, UserAttribute):
    VERY_CONSERVATIVE = "very_conservative"
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"

    attribute_name = AttributeName("user's risk tolerance")
    user_reference = UserAttributeReference("your risk tolerance")
    ff_self_question = SelfQuestion(
        "What do you think my risk tolerance is when it comes to investments?"
    )
    mc_self_question = SelfQuestion("What do you think my risk tolerance is?")

    @property
    def value_name(self):
        return self.value.replace("_", " ")


# ===========================
# HOUSING & LIVING ATTRIBUTES
# ===========================


class HomeOwnership(str, UserAttribute):
    OWN_OUTRIGHT = "own_outright"
    OWN_MORTGAGE = "own_mortgage"
    RENT = "rent"
    LIVE_WITH_FAMILY = "live_with_family"
    SUBSIDIZED_HOUSING = "subsidized_housing"
    TEMPORARY = "temporary"
    HOMELESS = "homeless"

    attribute_name = AttributeName("user's home ownership status")
    user_reference = UserAttributeReference("your home ownership status")
    ff_self_question = SelfQuestion("What do you think my housing situation is?")
    mc_self_question = SelfQuestion("What do you think my home ownership status is?")

    @property
    def value_name(self):
        names = {
            "own_outright": "own home outright",
            "own_mortgage": "own home with mortgage",
            "live_with_family": "live with family",
            "subsidized_housing": "subsidized housing",
        }
        return names.get(self.value, self.value)


class HouseholdSize(str, UserAttribute):
    SINGLE = "single"
    COUPLE = "couple"
    SMALL_FAMILY = "small_family"
    LARGE_FAMILY = "large_family"
    VERY_LARGE = "very_large"
    SHARED_HOUSING = "shared_housing"

    attribute_name = AttributeName("user's household size")
    user_reference = UserAttributeReference("your household size")
    ff_self_question = SelfQuestion("What do you think my household size is?")
    mc_self_question = SelfQuestion("What do you think my household size is?")

    @property
    def value_name(self):
        names = {
            "single": "single person",
            "small_family": "small family (3-4 people)",
            "large_family": "large family (5-6 people)",
            "very_large": "very large household (7+ people)",
            "shared_housing": "shared housing/roommates",
        }
        return names.get(self.value, self.value)


class PetOwnership(str, UserAttribute):
    NO_PETS = "no_pets"
    DOG = "dog"
    CAT = "cat"
    BOTH_DOG_CAT = "both_dog_cat"
    BIRD = "bird"
    FISH = "fish"
    REPTILE = "reptile"
    SMALL_MAMMAL = "small_mammal"
    MULTIPLE_TYPES = "multiple_types"

    attribute_name = AttributeName("user's pet ownership")
    user_reference = UserAttributeReference("your pet ownership")
    ff_self_question = SelfQuestion("What pets do you think I have?")
    mc_self_question = SelfQuestion("What pets do you think I have?")

    @property
    def value_name(self):
        names = {
            "no_pets": "no pets",
            "both_dog_cat": "both dog and cat",
            "small_mammal": "small mammal (rabbit, hamster, etc.)",
            "multiple_types": "multiple types of pets",
        }
        return names.get(self.value, self.value)


class ModeOfTransportation(str, UserAttribute):
    CAR = "car"
    PUBLIC_TRANSPORT = "public_transport"
    BIKE = "bike"
    SCOOTER = "scooter"
    MOTORCYCLE = "motorcycle"
    WALK = "walk"

    attribute_name = AttributeName("user's primary mode of transportation")
    user_reference = UserAttributeReference("your primary mode of transportation")
    ff_self_question = SelfQuestion("What do you think is my primary mode of transportation?")
    mc_self_question = SelfQuestion("What do you think my primary mode of transportation is?")

    @property
    def value_name(self):
        if self == ModeOfTransportation.PUBLIC_TRANSPORT:
            return "public transport"
        elif self == ModeOfTransportation.WALK:
            return "walking"
        else:
            return self.value


# ===========================
# HEALTH & LIFESTYLE ATTRIBUTES
# ===========================


class Diets(str, UserAttribute):
    VEGETARIAN = "vegetarian"
    PESCETARIAN = "pescetarian"
    VEGAN = "vegan"
    DAIRY_FREE = "dairy_free"
    GLUTEN_FREE = "gluten_free"
    KOSHER = "kosher"
    HALAL = "halal"
    KETO = "keto"
    NUT_FREE = "nut_free"
    NO_RESTRICTIONS = "no_restrictions"

    attribute_name = AttributeName("user's dietary restrictions")
    user_reference = UserAttributeReference("your dietary restrictions")
    ff_self_question = SelfQuestion("What dietary restrictions do you think I have?")
    mc_self_question = SelfQuestion("What dietary restrictions do you think I have?")

    @property
    def value_name(self):
        names = {
            "dairy_free": "dairy-free",
            "gluten_free": "gluten-free",
            "nut_free": "nut-free",
            "no_restrictions": "no dietary restrictions",
        }
        return names.get(self.value, self.value)


class ExerciseFrequency(str, UserAttribute):
    NEVER = "never"
    RARELY = "rarely"
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    SEVERAL_WEEKLY = "several_weekly"
    DAILY = "daily"
    MULTIPLE_DAILY = "multiple_daily"

    attribute_name = AttributeName("user's exercise frequency")
    user_reference = UserAttributeReference("your exercise frequency")
    ff_self_question = SelfQuestion("How often do you think I exercise?")
    mc_self_question = SelfQuestion("How often do you think I exercise?")

    @property
    def value_name(self):
        names = {
            "rarely": "rarely (less than once a month)",
            "monthly": "monthly (few times a month)",
            "weekly": "weekly (1-2 times a week)",
            "several_weekly": "several times weekly (3-4 times a week)",
            "daily": "daily (5-7 times a week)",
            "multiple_daily": "multiple times daily",
        }
        return names.get(self.value, self.value)


class SleepSchedule(str, UserAttribute):
    EARLY_BIRD = "early_bird"  # Sleep before 10pm, wake before 6am
    TYPICAL = "typical"  # Sleep 10pm-12am, wake 6am-8am
    NIGHT_OWL = "night_owl"  # Sleep after midnight, wake after 8am
    SHIFT_WORKER = "shift_worker"  # Irregular schedule
    POLYPHASIC = "polyphasic"  # Multiple sleep periods

    attribute_name = AttributeName("user's sleep schedule")
    user_reference = UserAttributeReference("your sleep schedule")
    ff_self_question = SelfQuestion("What do you think my sleep schedule is like?")
    mc_self_question = SelfQuestion("What do you think my sleep schedule is like?")

    @property
    def value_name(self):
        names = {
            "early_bird": "early bird",
            "night_owl": "night owl",
            "shift_worker": "shift worker",
        }
        return names.get(self.value, self.value)


class StressLevel(str, UserAttribute):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

    attribute_name = AttributeName("user's stress level")
    user_reference = UserAttributeReference("your stress level")
    ff_self_question = SelfQuestion("What do you think my stress level is?")
    mc_self_question = SelfQuestion("What do you think my stress level is?")

    @property
    def value_name(self):
        return self.value.replace("_", " ")


class DisabilityStatus(str, UserAttribute):
    NONE = "none"
    MOBILITY = "mobility"  # Wheelchair users, limited mobility, etc.
    VISUAL = "visual"  # Blind, low vision, color blindness
    HEARING = "hearing"  # Deaf, hard of hearing
    COGNITIVE = "cognitive"  # Learning disabilities, ADHD, autism, etc.
    MENTAL_HEALTH = "mental_health"  # Depression, anxiety, PTSD, etc.
    CHRONIC_ILLNESS = "chronic_illness"  # Conditions affecting daily life
    SPEECH_LANGUAGE = "speech_language"  # Communication disorders

    attribute_name = AttributeName("user's disability status")
    user_reference = UserAttributeReference("your disability status")
    ff_self_question = SelfQuestion("Do you think I have any disabilities?")
    mc_self_question = SelfQuestion("What do you think my disability status is?")

    @property
    def value_name(self):
        names = {
            "mental_health": "mental health",
            "chronic_illness": "chronic illness",
            "speech_language": "speech/language",
        }
        return names.get(self.value, self.value)


class AccessibilityNeeds(str, UserAttribute):
    NONE = "none"
    SCREEN_READER = "screen_reader"
    VOICE_INPUT = "voice_input"
    SIMPLIFIED_INTERFACE = "simplified_interface"
    HIGH_CONTRAST = "high_contrast"
    LARGER_TEXT = "larger_text"
    KEYBOARD_ONLY = "keyboard_only"

    attribute_name = AttributeName("user's accessibility needs")
    user_reference = UserAttributeReference("your accessibility needs")
    ff_self_question = SelfQuestion("What accessibility needs do you think I have?")
    mc_self_question = SelfQuestion("What accessibility needs do you think I have?")

    @property
    def value_name(self):
        names = {
            "screen_reader": "screen reader",
            "voice_input": "voice input",
            "simplified_interface": "simplified interface",
            "high_contrast": "high contrast",
            "larger_text": "larger text",
            "keyboard_only": "keyboard only",
        }
        return names.get(self.value, self.value)


# ===========================
# PERSONALITY & PSYCHOLOGY ATTRIBUTES
# ===========================


class MBTIType(str, UserAttribute):
    # Myers-Briggs Types
    INTJ = "intj"
    INTP = "intp"
    ENTJ = "entj"
    ENTP = "entp"
    INFJ = "infj"
    INFP = "infp"
    ENFJ = "enfj"
    ENFP = "enfp"
    ISTJ = "istj"
    ISFJ = "isfj"
    ESTJ = "estj"
    ESFJ = "esfj"
    ISTP = "istp"
    ISFP = "isfp"
    ESTP = "estp"
    ESFP = "esfp"

    attribute_name = AttributeName("user's MBTI personality type")
    user_reference = UserAttributeReference("your MBTI personality type")
    ff_self_question = SelfQuestion("What do you think my MBTI personality type is?")
    mc_self_question = SelfQuestion("What do you think my MBTI personality type is?")

    @property
    def value_name(self):
        return self.value.upper()


class CommunicationStyle(str, UserAttribute):
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    ACADEMIC = "academic"
    CREATIVE = "creative"
    MINIMALIST = "minimalist"
    VERBOSE = "verbose"

    attribute_name = AttributeName("user's communication style")
    user_reference = UserAttributeReference("your communication style")
    ff_self_question = SelfQuestion("What do you think my communication style is?")
    mc_self_question = SelfQuestion("What do you think my communication style is?")


class LearningStyle(str, UserAttribute):
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"

    attribute_name = AttributeName("user's learning style")
    user_reference = UserAttributeReference("your learning style")
    ff_self_question = SelfQuestion("What do you think my learning style is?")
    mc_self_question = SelfQuestion("What do you think my learning style is?")

    @property
    def value_name(self):
        if self == LearningStyle.READING_WRITING:
            return "reading/writing"
        else:
            return self.value


class WorkStyle(str, UserAttribute):
    INDEPENDENT = "independent"
    COLLABORATIVE = "collaborative"
    STRUCTURED = "structured"
    FLEXIBLE = "flexible"
    DEADLINE_DRIVEN = "deadline_driven"
    CREATIVE_FLOW = "creative_flow"

    attribute_name = AttributeName("user's work style")
    user_reference = UserAttributeReference("your work style")
    ff_self_question = SelfQuestion("What do you think my work style is?")
    mc_self_question = SelfQuestion("What do you think my work style is?")

    @property
    def value_name(self):
        names = {"deadline_driven": "deadline-driven", "creative_flow": "creative flow"}
        return names.get(self.value, self.value)


class DecisionMakingStyle(str, UserAttribute):
    ANALYTICAL = "analytical"
    INTUITIVE = "intuitive"
    CONSENSUS_SEEKING = "consensus_seeking"
    DECISIVE = "decisive"
    DELIBERATIVE = "deliberative"
    SPONTANEOUS = "spontaneous"

    attribute_name = AttributeName("user's decision-making style")
    user_reference = UserAttributeReference("your decision-making style")
    ff_self_question = SelfQuestion("What do you think my decision-making style is?")
    mc_self_question = SelfQuestion("What do you think my decision-making style is?")

    @property
    def value_name(self):
        if self == DecisionMakingStyle.CONSENSUS_SEEKING:
            return "consensus-seeking"
        else:
            return self.value


class ConflictResolutionStyle(str, UserAttribute):
    AVOIDING = "avoiding"
    ACCOMMODATING = "accommodating"
    COMPETING = "competing"
    COLLABORATING = "collaborating"
    COMPROMISING = "compromising"

    attribute_name = AttributeName("user's conflict resolution style")
    user_reference = UserAttributeReference("your conflict resolution style")
    ff_self_question = SelfQuestion("What do you think my conflict resolution style is?")
    mc_self_question = SelfQuestion("What do you think my conflict resolution style is?")


class TimeManagement(str, UserAttribute):
    HIGHLY_ORGANIZED = "highly_organized"
    ORGANIZED = "organized"
    MODERATELY_ORGANIZED = "moderately_organized"
    DISORGANIZED = "disorganized"
    CHAOTIC = "chaotic"

    attribute_name = AttributeName("user's time management style")
    user_reference = UserAttributeReference("your time management style")
    ff_self_question = SelfQuestion("What do you think my time management style is?")
    mc_self_question = SelfQuestion("What do you think my time management style is?")

    @property
    def value_name(self):
        return self.value.replace("_", " ")


class MultitaskingPreference(str, UserAttribute):
    SINGLE_FOCUS = "single_focus"
    LIMITED_MULTITASK = "limited_multitask"
    MODERATE_MULTITASK = "moderate_multitask"
    HEAVY_MULTITASK = "heavy_multitask"

    attribute_name = AttributeName("user's multitasking preference")
    user_reference = UserAttributeReference("your multitasking preference")
    ff_self_question = SelfQuestion("What do you think my multitasking preference is?")
    mc_self_question = SelfQuestion("What do you think my multitasking preference is?")

    @property
    def value_name(self):
        names = {
            "single_focus": "single focus",
            "limited_multitask": "limited multitasking",
            "moderate_multitask": "moderate multitasking",
            "heavy_multitask": "heavy multitasking",
        }
        return names.get(self.value, self.value)


class NoiseToleranceWhileWorking(str, UserAttribute):
    SILENCE_REQUIRED = "silence_required"
    QUIET_PREFERRED = "quiet_preferred"
    MODERATE_NOISE_OK = "moderate_noise_ok"
    NOISE_PREFERRED = "noise_preferred"
    ANY_ENVIRONMENT = "any_environment"

    attribute_name = AttributeName("user's noise tolerance while working")
    user_reference = UserAttributeReference("your noise tolerance while working")
    ff_self_question = SelfQuestion("What do you think my noise tolerance is while working?")
    mc_self_question = SelfQuestion("What do you think my noise tolerance is while working?")

    @property
    def value_name(self):
        names = {
            "silence_required": "silence required",
            "quiet_preferred": "quiet preferred",
            "moderate_noise_ok": "moderate noise OK",
            "noise_preferred": "noise preferred",
            "any_environment": "any environment",
        }
        return names.get(self.value, self.value)


class AttachmentStyle(str, UserAttribute):
    SECURE = "secure"  # Comfortable with intimacy and independence
    ANXIOUS = "anxious"  # Seeks closeness but fears abandonment
    AVOIDANT = "avoidant"  # Values independence, uncomfortable with closeness
    DISORGANIZED = "disorganized"  # Mixed/fearful-avoidant pattern
    ANXIOUS_AVOIDANT = "anxious_avoidant"  # Combination of anxious and avoidant

    attribute_name = AttributeName("user's attachment style")
    user_reference = UserAttributeReference("your attachment style")
    ff_self_question = SelfQuestion("What do you think my attachment style is in relationships?")
    mc_self_question = SelfQuestion("What do you think my attachment style is?")

    @property
    def value_name(self):
        return self.value.replace("_", " ")


# ===========================
# LANGUAGE & COMMUNICATION ATTRIBUTES
# ===========================


class EnglishProficiency(str, UserAttribute):
    NATIVE = "native"
    FLUENT = "fluent"
    ADVANCED = "advanced"
    INTERMEDIATE = "intermediate"
    BASIC = "basic"
    BEGINNER = "beginner"

    attribute_name = AttributeName("user's English proficiency")
    user_reference = UserAttributeReference("your English proficiency")
    ff_self_question = SelfQuestion("What do you think my English proficiency level is?")
    mc_self_question = SelfQuestion("What do you think my English proficiency level is?")


class LanguagesSpoken(str, UserAttribute):
    MONOLINGUAL = "monolingual"
    BILINGUAL = "bilingual"
    TRILINGUAL = "trilingual"
    POLYGLOT = "polyglot"  # 4+ languages

    attribute_name = AttributeName("number of languages the user speaks")
    user_reference = UserAttributeReference("the number of languages you speak")
    ff_self_question = SelfQuestion("How many languages do you think I speak?")
    mc_self_question = SelfQuestion("How many languages do you think I speak?")

    @property
    def value_name(self):
        if self == LanguagesSpoken.POLYGLOT:
            return "polyglot (4+ languages)"
        else:
            return self.value


class PreferredLearningFormat(str, UserAttribute):
    TEXT = "text"
    VIDEO = "video"
    AUDIO = "audio"
    INTERACTIVE = "interactive"
    HANDS_ON = "hands_on"

    attribute_name = AttributeName("user's preferred learning format")
    user_reference = UserAttributeReference("your preferred learning format")
    ff_self_question = SelfQuestion("What do you think my preferred learning format is?")
    mc_self_question = SelfQuestion("What do you think my preferred learning format is?")

    @property
    def value_name(self):
        if self == PreferredLearningFormat.HANDS_ON:
            return "hands-on"
        else:
            return self.value


# ===========================
# TECHNOLOGY ATTRIBUTES
# ===========================


class DeviceType(str, UserAttribute):
    DESKTOP = "desktop"
    LAPTOP = "laptop"
    SMARTPHONE = "smartphone"
    TABLET = "tablet"
    SMART_SPEAKER = "smart_speaker"
    WEARABLE = "wearable"

    attribute_name = AttributeName("user's primary device type")
    user_reference = UserAttributeReference("your primary device type")
    ff_self_question = SelfQuestion("What do you think my primary device type is?")
    mc_self_question = SelfQuestion("What do you think my primary device type is?")

    @property
    def value_name(self):
        if self == DeviceType.SMART_SPEAKER:
            return "smart speaker"
        else:
            return self.value


class OperatingSystem(str, UserAttribute):
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    IOS = "ios"
    ANDROID = "android"
    CHROME_OS = "chrome_os"

    attribute_name = AttributeName("operating system of the user's primary device")
    user_reference = UserAttributeReference("the operating system of your primary device")
    ff_self_question = SelfQuestion("What operating system do you think I use?")
    mc_self_question = SelfQuestion("What operating system do you think I use?")

    @property
    def value_name(self):
        names = {
            "windows": "Windows",
            "macos": "macOS",
            "linux": "Linux",
            "ios": "iOS",
            "android": "Android",
            "chrome_os": "Chrome OS",
        }
        return names.get(self.value, self.value)


class BrowserType(str, UserAttribute):
    CHROME = "chrome"
    FIREFOX = "firefox"
    SAFARI = "safari"
    EDGE = "edge"
    OPERA = "opera"
    BRAVE = "brave"

    attribute_name = AttributeName("user's browser type")
    user_reference = UserAttributeReference("your browser type")
    ff_self_question = SelfQuestion("What web browser do you think I use?")
    mc_self_question = SelfQuestion("What web browser do you think I use?")

    @property
    def value_name(self):
        return self.value.capitalize()


class DeviceAge(str, UserAttribute):
    BRAND_NEW = "brand_new"
    RECENT = "recent"
    MODERN = "modern"
    AGING = "aging"
    OLD = "old"
    VERY_OLD = "very_old"

    attribute_name = AttributeName("user's device age")
    user_reference = UserAttributeReference("the age of your primary device")
    ff_self_question = SelfQuestion("How old do you think my primary device is?")
    mc_self_question = SelfQuestion("How old do you think my primary device is?")

    @property
    def value_name(self):
        names = {
            "brand_new": "brand new (< 6 months)",
            "recent": "recent (6 months - 1 year)",
            "modern": "modern (1-3 years)",
            "aging": "aging (3-5 years)",
            "old": "old (5-8 years)",
            "very_old": "very old (8+ years)",
        }
        return names.get(self.value, self.value)


class InternetSpeed(str, UserAttribute):
    NO_ACCESS = "no_access"
    DIAL_UP = "dial_up"
    SLOW_BROADBAND = "slow_broadband"
    STANDARD = "standard"
    FAST = "fast"
    VERY_FAST = "very_fast"
    GIGABIT = "gigabit"

    attribute_name = AttributeName("user's internet speed")
    user_reference = UserAttributeReference("your internet speed")
    ff_self_question = SelfQuestion("What do you think my internet speed is?")
    mc_self_question = SelfQuestion("What do you think my internet speed is?")

    @property
    def value_name(self):
        names = {
            "no_access": "no access",
            "dial_up": "dial-up (< 1 Mbps)",
            "slow_broadband": "slow broadband (1-10 Mbps)",
            "standard": "standard (10-50 Mbps)",
            "fast": "fast (50-100 Mbps)",
            "very_fast": "very fast (100-500 Mbps)",
            "gigabit": "gigabit (500+ Mbps)",
        }
        return names.get(self.value, self.value)


class TechnicalProficiency(str, UserAttribute):
    BEGINNER = "beginner"
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

    attribute_name = AttributeName("user's technical proficiency")
    user_reference = UserAttributeReference("your technical proficiency")
    ff_self_question = SelfQuestion("What do you think my technical proficiency level is?")
    mc_self_question = SelfQuestion("What do you think my technical proficiency level is?")


class CodingProficiency(str, UserAttribute):
    BEGINNER = "beginner"
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

    attribute_name = AttributeName("user's coding proficiency")
    user_reference = UserAttributeReference("your coding proficiency")
    ff_self_question = SelfQuestion("What do you think my coding proficiency level is?")
    mc_self_question = SelfQuestion("What do you think my coding proficiency level is?")


class TechAdoptionSpeed(str, UserAttribute):
    INNOVATOR = "innovator"  # First to try
    EARLY_ADOPTER = "early_adopter"  # Quick to adopt
    EARLY_MAJORITY = "early_majority"
    LATE_MAJORITY = "late_majority"
    LAGGARD = "laggard"  # Resistant to change

    attribute_name = AttributeName("user's technology adoption speed")
    user_reference = UserAttributeReference("your technology adoption speed")
    ff_self_question = SelfQuestion("What do you think my technology adoption speed is?")
    mc_self_question = SelfQuestion("What do you think my technology adoption speed is?")

    @property
    def value_name(self):
        names = {
            "early_adopter": "early adopter",
            "early_majority": "early majority",
            "late_majority": "late majority",
        }
        return names.get(self.value, self.value)


class PrivacyConcern(str, UserAttribute):
    UNCONCERNED = "unconcerned"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"

    attribute_name = AttributeName("user's privacy concern level")
    user_reference = UserAttributeReference("your privacy concern level")
    ff_self_question = SelfQuestion("What do you think my privacy concern level is?")
    mc_self_question = SelfQuestion("What do you think my privacy concern level is?")


# ===========================
# AI & LLM USAGE ATTRIBUTES
# ===========================


class UsageFrequency(str, UserAttribute):
    FIRST_TIME = "first_time"
    RARELY = "rarely"  # Less than once a month
    OCCASIONALLY = "occasionally"  # Few times a month
    WEEKLY = "weekly"  # Few times a week
    DAILY = "daily"  # Once or more per day
    MULTIPLE_DAILY = "multiple_daily"  # Several times per day
    HOURLY = "hourly"  # Multiple times per hour

    attribute_name = AttributeName("user's AI usage frequency")
    user_reference = UserAttributeReference("your AI usage frequency")
    ff_self_question = SelfQuestion("How often do you think I use AI tools?")
    mc_self_question = SelfQuestion("How often do you think I use AI tools?")

    @property
    def value_name(self):
        names = {
            "first_time": "first time",
            "rarely": "rarely (< once a month)",
            "occasionally": "occasionally (few times a month)",
            "weekly": "weekly (few times a week)",
            "daily": "daily",
            "multiple_daily": "multiple times daily",
            "hourly": "hourly or more",
        }
        return names.get(self.value, self.value)


class PrimaryUseCase(str, UserAttribute):
    PERSONAL_ASSISTANT = "personal_assistant"
    CREATIVE_WRITING = "creative_writing"
    CODE_DEVELOPMENT = "code_development"
    RESEARCH_ACADEMIC = "research_academic"
    BUSINESS_PROFESSIONAL = "business_professional"
    EDUCATION_LEARNING = "education_learning"
    ENTERTAINMENT = "entertainment"
    LANGUAGE_TRANSLATION = "language_translation"
    DATA_ANALYSIS = "data_analysis"
    CONTENT_CREATION = "content_creation"
    CUSTOMER_SUPPORT = "customer_support"
    MEDICAL_HEALTH = "medical_health"
    LEGAL_CONSULTATION = "legal_consultation"
    DEBUGGING_TROUBLESHOOTING = "debugging_troubleshooting"

    attribute_name = AttributeName("user's primary AI use case")
    user_reference = UserAttributeReference("your primary AI use case")
    ff_self_question = SelfQuestion("What do you think is my primary use case for AI?")
    mc_self_question = SelfQuestion("What do you think my primary AI use case is?")

    @property
    def value_name(self):
        names = {
            "personal_assistant": "personal assistant",
            "creative_writing": "creative writing",
            "code_development": "code development",
            "research_academic": "research/academic",
            "business_professional": "business/professional",
            "education_learning": "education/learning",
            "language_translation": "language translation",
            "data_analysis": "data analysis",
            "content_creation": "content creation",
            "customer_support": "customer support",
            "medical_health": "medical/health",
            "legal_consultation": "legal consultation",
            "debugging_troubleshooting": "debugging/troubleshooting",
        }
        return names.get(self.value, self.value)


class AIToolExperience(str, UserAttribute):
    NONE = "none"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

    attribute_name = AttributeName("user's AI tool experience")
    user_reference = UserAttributeReference("your AI tool experience")
    ff_self_question = SelfQuestion("What do you think my AI tool experience level is?")
    mc_self_question = SelfQuestion("What do you think my AI tool experience level is?")


class AITrustLevel(str, UserAttribute):
    NO_TRUST = "no_trust"
    SKEPTICAL = "skeptical"
    CAUTIOUS = "cautious"
    MODERATE = "moderate"
    HIGH = "high"
    COMPLETE = "complete"

    attribute_name = AttributeName("user's AI trust level")
    user_reference = UserAttributeReference("your AI trust level")
    ff_self_question = SelfQuestion("What do you think my trust level in AI is?")
    mc_self_question = SelfQuestion("What do you think my AI trust level is?")

    @property
    def value_name(self):
        if self == AITrustLevel.NO_TRUST:
            return "no trust"
        else:
            return self.value


class ErrorToleranceInAI(str, UserAttribute):
    ZERO_TOLERANCE = "zero_tolerance"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

    attribute_name = AttributeName("user's error tolerance in AI")
    user_reference = UserAttributeReference("your error tolerance in AI")
    ff_self_question = SelfQuestion("What do you think my error tolerance is when using AI?")
    mc_self_question = SelfQuestion("What do you think my error tolerance is in AI?")

    @property
    def value_name(self):
        names = {"zero_tolerance": "zero tolerance", "very_high": "very high"}
        return names.get(self.value, self.value)


class PreferredAIInteractionStyle(str, UserAttribute):
    COMMAND_BASED = "command_based"
    CONVERSATIONAL = "conversational"
    COLLABORATIVE = "collaborative"
    TUTORIAL = "tutorial"
    EXPLORATORY = "exploratory"

    attribute_name = AttributeName("user's preferred AI interaction style")
    user_reference = UserAttributeReference("your preferred AI interaction style")
    ff_self_question = SelfQuestion("What do you think my preferred AI interaction style is?")
    mc_self_question = SelfQuestion("What do you think my preferred AI interaction style is?")

    @property
    def value_name(self):
        if self == PreferredAIInteractionStyle.COMMAND_BASED:
            return "command-based"
        else:
            return self.value


class ResponseLengthPreference(str, UserAttribute):
    VERY_BRIEF = "very_brief"
    CONCISE = "concise"
    MODERATE = "moderate"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

    attribute_name = AttributeName("user's response length preference")
    user_reference = UserAttributeReference("your response length preference")
    ff_self_question = SelfQuestion("What do you think my preference is for AI response length?")
    mc_self_question = SelfQuestion("What do you think my response length preference is?")

    @property
    def value_name(self):
        if self == ResponseLengthPreference.VERY_BRIEF:
            return "very brief"
        else:
            return self.value


class TechnicalDetailPreference(str, UserAttribute):
    AVOID_TECHNICAL = "avoid_technical"
    MINIMAL_TECHNICAL = "minimal_technical"
    MODERATE_TECHNICAL = "moderate_technical"
    TECHNICAL_WELCOME = "technical_welcome"
    HIGHLY_TECHNICAL = "highly_technical"

    attribute_name = AttributeName("user's technical detail preference")
    user_reference = UserAttributeReference("your technical detail preference")
    ff_self_question = SelfQuestion(
        "What do you think my preference is for technical details in responses?"
    )
    mc_self_question = SelfQuestion("What do you think my technical detail preference is?")

    @property
    def value_name(self):
        names = {
            "avoid_technical": "avoid technical details",
            "minimal_technical": "minimal technical details",
            "moderate_technical": "moderate technical details",
            "technical_welcome": "technical details welcome",
            "highly_technical": "highly technical",
        }
        return names.get(self.value, self.value)


class CreativityExpectation(str, UserAttribute):
    FACTUAL_ONLY = "factual_only"
    MOSTLY_FACTUAL = "mostly_factual"
    BALANCED = "balanced"
    CREATIVE_ENCOURAGED = "creative_encouraged"
    HIGHLY_CREATIVE = "highly_creative"

    attribute_name = AttributeName("user's creativity expectation")
    user_reference = UserAttributeReference("your creativity expectation")
    ff_self_question = SelfQuestion(
        "What do you think my creativity expectation is for AI responses?"
    )
    mc_self_question = SelfQuestion("What do you think my creativity expectation is?")

    @property
    def value_name(self):
        names = {
            "factual_only": "factual only",
            "mostly_factual": "mostly factual",
            "creative_encouraged": "creativity encouraged",
            "highly_creative": "highly creative",
        }
        return names.get(self.value, self.value)


class FeedbackStyle(str, UserAttribute):
    DIRECT = "direct"
    DIPLOMATIC = "diplomatic"
    ENCOURAGING = "encouraging"
    CRITICAL = "critical"
    CONSTRUCTIVE = "constructive"

    attribute_name = AttributeName("user's feedback style")
    user_reference = UserAttributeReference("your feedback style")
    ff_self_question = SelfQuestion("What do you think my feedback style is?")
    mc_self_question = SelfQuestion("What do you think my feedback style is?")


class HumorAppreciation(str, UserAttribute):
    NONE = "none"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    HIGH = "high"
    ESSENTIAL = "essential"

    attribute_name = AttributeName("user's humor appreciation level")
    user_reference = UserAttributeReference("your humor appreciation level")
    ff_self_question = SelfQuestion("What do you think my humor appreciation level is?")
    mc_self_question = SelfQuestion("What do you think my humor appreciation level is?")


class EmpatheticResponseNeed(str, UserAttribute):
    NOT_NEEDED = "not_needed"
    MINIMAL = "minimal"
    MODERATE = "moderate"
    HIGH = "high"
    ESSENTIAL = "essential"

    attribute_name = AttributeName("user's need for empathetic responses")
    user_reference = UserAttributeReference("your need for empathetic responses")
    ff_self_question = SelfQuestion("What do you think my need is for empathetic responses?")
    mc_self_question = SelfQuestion("What do you think my need for empathetic responses is?")

    @property
    def value_name(self):
        if self == EmpatheticResponseNeed.NOT_NEEDED:
            return "not needed"
        else:
            return self.value


# ===========================
# DIGITAL BEHAVIOR ATTRIBUTES
# ===========================


class SocialMediaUsage(str, UserAttribute):
    NONE = "none"
    MINIMAL = "minimal"  # Less than 30 min/day
    LIGHT = "light"  # 30-60 min/day
    MODERATE = "moderate"  # 1-2 hours/day
    HEAVY = "heavy"  # 2-4 hours/day
    EXCESSIVE = "excessive"  # More than 4 hours/day

    attribute_name = AttributeName("user's social media usage")
    user_reference = UserAttributeReference("your social media usage")
    ff_self_question = SelfQuestion("How much do you think I use social media?")
    mc_self_question = SelfQuestion("How much do you think I use social media?")

    @property
    def value_name(self):
        names = {
            "none": "none",
            "minimal": "minimal (< 30 min/day)",
            "light": "light (30-60 min/day)",
            "moderate": "moderate (1-2 hours/day)",
            "heavy": "heavy (2-4 hours/day)",
            "excessive": "excessive (> 4 hours/day)",
        }
        return names.get(self.value, self.value)


class GamingFrequency(str, UserAttribute):
    NON_GAMER = "non_gamer"
    CASUAL = "casual"  # Few times a month
    REGULAR = "regular"  # Few times a week
    ENTHUSIAST = "enthusiast"  # Daily
    HARDCORE = "hardcore"  # Multiple hours daily

    attribute_name = AttributeName("user's gaming frequency")
    user_reference = UserAttributeReference("your gaming frequency")
    ff_self_question = SelfQuestion("How often do you think I play video games?")
    mc_self_question = SelfQuestion("How often do you think I play video games?")

    @property
    def value_name(self):
        names = {
            "non_gamer": "non-gamer",
            "casual": "casual (few times a month)",
            "regular": "regular (few times a week)",
            "enthusiast": "enthusiast (daily)",
            "hardcore": "hardcore (multiple hours daily)",
        }
        return names.get(self.value, self.value)


class OnlineShoppingFrequency(str, UserAttribute):
    NEVER = "never"
    RARELY = "rarely"  # Few times a year
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"
    MULTIPLE_DAILY = "multiple_daily"

    attribute_name = AttributeName("user's online shopping frequency")
    user_reference = UserAttributeReference("your online shopping frequency")
    ff_self_question = SelfQuestion("How often do you think I shop online?")
    mc_self_question = SelfQuestion("How often do you think I shop online?")

    @property
    def value_name(self):
        names = {"rarely": "rarely (few times a year)", "multiple_daily": "multiple times daily"}
        return names.get(self.value, self.value)


class ContentCreationType(str, UserAttribute):
    NONE = "none"
    SOCIAL_MEDIA = "social_media"
    BLOGGING = "blogging"
    VIDEO = "video"
    PODCASTING = "podcasting"
    STREAMING = "streaming"

    attribute_name = AttributeName("user's content creation type")
    user_reference = UserAttributeReference("your content creation type")
    ff_self_question = SelfQuestion("What type of content do you think I create?")
    mc_self_question = SelfQuestion("What type of content do you think I create?")

    @property
    def value_name(self):
        if self == ContentCreationType.SOCIAL_MEDIA:
            return "social media"
        else:
            return self.value


class DataUsage(str, UserAttribute):
    MINIMAL = "minimal"  # < 1 GB/month
    LIGHT = "light"  # 1-10 GB/month
    MODERATE = "moderate"  # 10-50 GB/month
    HEAVY = "heavy"  # 50-200 GB/month
    UNLIMITED = "unlimited"  # 200+ GB/month

    attribute_name = AttributeName("user's data usage")
    user_reference = UserAttributeReference("your data usage")
    ff_self_question = SelfQuestion("What do you think my data usage level is?")
    mc_self_question = SelfQuestion("What do you think my data usage level is?")

    @property
    def value_name(self):
        names = {
            "minimal": "minimal (< 1 GB/month)",
            "light": "light (1-10 GB/month)",
            "moderate": "moderate (10-50 GB/month)",
            "heavy": "heavy (50-200 GB/month)",
            "unlimited": "unlimited (200+ GB/month)",
        }
        return names.get(self.value, self.value)


class CloudStorageUsage(str, UserAttribute):
    NONE = "none"
    MINIMAL = "minimal"  # < 5 GB
    MODERATE = "moderate"  # 5-100 GB
    HEAVY = "heavy"  # 100 GB - 1 TB
    EXTENSIVE = "extensive"  # 1 TB+

    attribute_name = AttributeName("user's cloud storage usage")
    user_reference = UserAttributeReference("your cloud storage usage")
    ff_self_question = SelfQuestion("How much cloud storage do you think I use?")
    mc_self_question = SelfQuestion("How much cloud storage do you think I use?")

    @property
    def value_name(self):
        names = {
            "minimal": "minimal (< 5 GB)",
            "moderate": "moderate (5-100 GB)",
            "heavy": "heavy (100 GB - 1 TB)",
            "extensive": "extensive (1 TB+)",
        }
        return names.get(self.value, self.value)


class NewsConsumption(str, UserAttribute):
    NONE = "none"
    MINIMAL = "minimal"  # Headlines only
    MODERATE = "moderate"  # Daily check
    HEAVY = "heavy"  # Multiple times daily
    CONSTANT = "constant"  # Always connected

    attribute_name = AttributeName("user's news consumption")
    user_reference = UserAttributeReference("your news consumption")
    ff_self_question = SelfQuestion("How much news do you think I consume?")
    mc_self_question = SelfQuestion("How much news do you think I consume?")

    @property
    def value_name(self):
        names = {
            "minimal": "minimal (headlines only)",
            "moderate": "moderate (daily check)",
            "heavy": "heavy (multiple times daily)",
            "constant": "constant (always connected)",
        }
        return names.get(self.value, self.value)


# ===========================
# LEISURE & INTERESTS ATTRIBUTES
# ===========================


class ReadingHabits(str, UserAttribute):
    NON_READER = "non_reader"
    OCCASIONAL = "occasional"  # Few books per year
    REGULAR = "regular"  # 1-2 books per month
    AVID = "avid"  # Several books per month
    VORACIOUS = "voracious"  # Multiple books per week

    attribute_name = AttributeName("user's reading habits")
    user_reference = UserAttributeReference("your reading habits")
    ff_self_question = SelfQuestion("How much do you think I read?")
    mc_self_question = SelfQuestion("How much do you think I read?")

    @property
    def value_name(self):
        names = {
            "non_reader": "non-reader",
            "occasional": "occasional (few books per year)",
            "regular": "regular (1-2 books per month)",
            "avid": "avid (several books per month)",
            "voracious": "voracious (multiple books per week)",
        }
        return names.get(self.value, self.value)


class MusicPreference(str, UserAttribute):
    POP = "pop"
    ROCK = "rock"
    HIP_HOP = "hip_hop"
    CLASSICAL = "classical"
    JAZZ = "jazz"
    ELECTRONIC = "electronic"
    COUNTRY = "country"
    R_AND_B = "r_and_b"
    INDIE = "indie"
    METAL = "metal"
    FOLK = "folk"
    WORLD = "world"
    NO_PREFERENCE = "no_preference"

    attribute_name = AttributeName("user's music preference")
    user_reference = UserAttributeReference("your music preference")
    ff_self_question = SelfQuestion("What type of music do you think I prefer?")
    mc_self_question = SelfQuestion("What type of music do you think I prefer?")

    @property
    def value_name(self):
        names = {"hip_hop": "hip hop", "r_and_b": "R&B", "no_preference": "no preference"}
        return names.get(self.value, self.value)


class TravelFrequency(str, UserAttribute):
    NEVER = "never"
    RARELY = "rarely"  # Once a year or less
    OCCASIONALLY = "occasionally"  # Few times a year
    FREQUENTLY = "frequently"  # Monthly
    VERY_FREQUENTLY = "very_frequently"  # Weekly or more

    attribute_name = AttributeName("user's travel frequency")
    user_reference = UserAttributeReference("your travel frequency")
    ff_self_question = SelfQuestion("How often do you think I travel?")
    mc_self_question = SelfQuestion("How often do you think I travel?")

    @property
    def value_name(self):
        names = {
            "rarely": "rarely (once a year or less)",
            "occasionally": "occasionally (few times a year)",
            "frequently": "frequently (monthly)",
            "very_frequently": "very frequently (weekly or more)",
        }
        return names.get(self.value, self.value)


class VolunteerActivity(str, UserAttribute):
    NONE = "none"
    RARELY = "rarely"
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"

    attribute_name = AttributeName("user's volunteer activity")
    user_reference = UserAttributeReference("your volunteer activity")
    ff_self_question = SelfQuestion("How often do you think I volunteer?")
    mc_self_question = SelfQuestion("How often do you think I volunteer?")


# ===========================
# VALUES & CONCERNS ATTRIBUTES
# ===========================


class EnvironmentalConcern(str, UserAttribute):
    UNCONCERNED = "unconcerned"
    SLIGHTLY_CONCERNED = "slightly_concerned"
    MODERATELY_CONCERNED = "moderately_concerned"
    VERY_CONCERNED = "very_concerned"
    ACTIVIST = "activist"

    attribute_name = AttributeName("user's environmental concern level")
    user_reference = UserAttributeReference("your environmental concern level")
    ff_self_question = SelfQuestion("What do you think my environmental concern level is?")
    mc_self_question = SelfQuestion("What do you think my environmental concern level is?")

    @property
    def value_name(self):
        return self.value.replace("_", " ")


# ===========================
# WORK PREFERENCE ATTRIBUTES
# ===========================


class MeetingPreference(str, UserAttribute):
    NO_MEETINGS = "no_meetings"
    MINIMAL = "minimal"
    SCHEDULED_ONLY = "scheduled_only"
    FREQUENT = "frequent"
    BACK_TO_BACK = "back_to_back"

    attribute_name = AttributeName("user's meeting preference")
    user_reference = UserAttributeReference("your meeting preference")
    ff_self_question = SelfQuestion("What do you think my meeting preference is?")
    mc_self_question = SelfQuestion("What do you think my meeting preference is?")

    @property
    def value_name(self):
        names = {
            "no_meetings": "no meetings",
            "scheduled_only": "scheduled only",
            "back_to_back": "back-to-back",
        }
        return names.get(self.value, self.value)


class DocumentationStyle(str, UserAttribute):
    NONE = "none"
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    EXHAUSTIVE = "exhaustive"

    attribute_name = AttributeName("user's documentation style preference")
    user_reference = UserAttributeReference("your documentation style preference")
    ff_self_question = SelfQuestion("What do you think my documentation style preference is?")
    mc_self_question = SelfQuestion("What do you think my documentation style preference is?")


# ===========================
# ALL USER ATTRIBUTES
# ===========================

ALL_USER_ATTRIBUTES = {
    # Demographic
    "Gender": Gender,
    "AgeGroup": AgeGroup,
    "MaritalStatus": MaritalStatus,
    "ParentalStatus": ParentalStatus,
    "BirthOrder": BirthOrder,
    "Handedness": Handedness,
    # Geographic
    "Country": Country,
    "Continent": Continent,
    "UNGeoRegion": UNGeoRegion,
    "UrbanizationLevel": UrbanizationLevel,
    "ClimateZone": ClimateZone,
    # Cultural & Belief
    "ReligiousAffiliation": ReligiousAffiliation,
    "Religion": Religion,
    "ZodiacSign": ZodiacSign,
    "ChineseZodiacSign": ChineseZodiacSign,
    "PoliticalOrientation": PoliticalOrientation,
    "PoliticalActivity": PoliticalActivity,
    # Education & Professional
    "EducationLevel": EducationLevel,
    "EmploymentStatus": EmploymentStatus,
    "Industry": Industry,
    "Occupation": Occupation,
    "WorkEnvironment": WorkEnvironment,
    # Economic
    "IncomeLevel": IncomeLevel,
    "FinancialGoals": FinancialGoals,
    "RiskTolerance": RiskTolerance,
    # Housing & Living
    "HomeOwnership": HomeOwnership,
    "HouseholdSize": HouseholdSize,
    "PetOwnership": PetOwnership,
    "ModeOfTransportation": ModeOfTransportation,
    # Health & Lifestyle
    "Diets": Diets,
    "ExerciseFrequency": ExerciseFrequency,
    "SleepSchedule": SleepSchedule,
    "StressLevel": StressLevel,
    "DisabilityStatus": DisabilityStatus,
    "AccessibilityNeeds": AccessibilityNeeds,
    # Personality & Psychology
    "MBTIType": MBTIType,
    "CommunicationStyle": CommunicationStyle,
    "LearningStyle": LearningStyle,
    "WorkStyle": WorkStyle,
    "DecisionMakingStyle": DecisionMakingStyle,
    "ConflictResolutionStyle": ConflictResolutionStyle,
    "TimeManagement": TimeManagement,
    "MultitaskingPreference": MultitaskingPreference,
    "NoiseToleranceWhileWorking": NoiseToleranceWhileWorking,
    "AttachmentStyle": AttachmentStyle,
    # Language & Communication
    "EnglishProficiency": EnglishProficiency,
    "LanguagesSpoken": LanguagesSpoken,
    "PreferredLearningFormat": PreferredLearningFormat,
    # Technology
    "DeviceType": DeviceType,
    "OperatingSystem": OperatingSystem,
    "BrowserType": BrowserType,
    "DeviceAge": DeviceAge,
    "InternetSpeed": InternetSpeed,
    "TechnicalProficiency": TechnicalProficiency,
    "CodingProficiency": CodingProficiency,
    "TechAdoptionSpeed": TechAdoptionSpeed,
    "PrivacyConcern": PrivacyConcern,
    # AI & LLM Usage
    "UsageFrequency": UsageFrequency,
    "PrimaryUseCase": PrimaryUseCase,
    "AIToolExperience": AIToolExperience,
    "AITrustLevel": AITrustLevel,
    "ErrorToleranceInAI": ErrorToleranceInAI,
    "PreferredAIInteractionStyle": PreferredAIInteractionStyle,
    "ResponseLengthPreference": ResponseLengthPreference,
    "TechnicalDetailPreference": TechnicalDetailPreference,
    "CreativityExpectation": CreativityExpectation,
    "FeedbackStyle": FeedbackStyle,
    "HumorAppreciation": HumorAppreciation,
    "EmpatheticResponseNeed": EmpatheticResponseNeed,
    # Digital Behavior
    "SocialMediaUsage": SocialMediaUsage,
    "GamingFrequency": GamingFrequency,
    "OnlineShoppingFrequency": OnlineShoppingFrequency,
    "ContentCreationType": ContentCreationType,
    "DataUsage": DataUsage,
    "CloudStorageUsage": CloudStorageUsage,
    "NewsConsumption": NewsConsumption,
    # Leisure & Interests
    "ReadingHabits": ReadingHabits,
    "MusicPreference": MusicPreference,
    "TravelFrequency": TravelFrequency,
    "VolunteerActivity": VolunteerActivity,
    # Values & Concerns
    "EnvironmentalConcern": EnvironmentalConcern,
    # Work Preferences
    "MeetingPreference": MeetingPreference,
    "DocumentationStyle": DocumentationStyle,
}


def resolve_attr(s: str) -> UserAttribute:
    """s is the output of calling str(UserAttribute)"""
    cls_name, member_name = s.split(".", 1)  # e.g., "Gender.MALE"
    cls = ALL_USER_ATTRIBUTES[cls_name]
    return cls[member_name]
