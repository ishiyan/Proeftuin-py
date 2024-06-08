// This program generates `mic.py`, overwriting any existing versions.
// It should be invoked by running `go run generate_mics_python` in this directory.

package main

import (
	"bytes"
	"encoding/csv"
	"errors"
	"fmt"
	"go/format"
	"io"
	"os"
	"sort"
	"strings"
	"time"
	"unicode"
)

const dataPublicationDate = "10-May-2021"

type market struct {
	country       string
	countryInc    string // the name of the country from the countries file
	code          string // ISO 3166 country code
	mic           string // market segment MIC
	micOp         string // operating MIC
	isOperational bool   // if an operational MIC or a segment MIC
	name          string // name-institution description
	acronym       string
	city          string
	website       string
	statusDate    string
	status        string
	creationDate  string
	comments      string
	tzsec         int // seconds east of UTC
}

// Contains an operational MIC with its optional segment MICs.
type mic struct {
	oper *market   // operational MIC
	segs []*market // segment MICs if any
}

// Contains a time zone offset in seconds and MICs having this offset.
type tzmic struct {
	tzsec   int // seconds east of UTC
	markets []*market
}

//nolint:misspell
const cxanMic = "CXAN"

var errUnknownTimezoneCity = errors.New("please add time zone for unknown city")

func main() {
	// selected country code slice and country code -> country name map
	cs, cm, err := readContries("countries.csv")
	if err != nil {
		die(err)
	}

	// a slice of all market pointers
	ms, err := readMarkets("ISO10383_MIC." + dataPublicationDate + ".csv")
	if err != nil {
		die(err)
	}

	// countries included from code generation
	ics := collectIncludedCountries(ms, cm)

	// countries excluded from code generation
	ecs := collectExcludedCountries(ms, cm)

	// enrich with time zone seconds and more friendly country names
	if err := enrichMarkets(ms, cm); err != nil {
		die(err)
	}

	// markets grouped by segments and arranged by country
	ams := arrangeByCountry(ms, cs)

	// markets grouped by seconds east of UTC
	tzmics := arrangeByTimeZone(ams)

	printMicsPython("mics.py", ams, ics, ecs, tzmics)
}

func readContries(filename string) ([]string, map[string]string, error) {
	cs := []string{}
	cm := map[string]string{}

	f, err := os.Open(filename)
	if err != nil {
		return cs, cm, fmt.Errorf("opening countries: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Comma = '|'
	r.Comment = '#'
	r.FieldsPerRecord = 2
	r.ReuseRecord = false
	r.TrimLeadingSpace = true

	ln := 0

	for {
		record, err := r.Read()
		if errors.Is(err, io.EOF) {
			break
		}

		ln++
		if err != nil {
			return cs, cm, fmt.Errorf("'%v' line %v: error reading file: %w", filename, ln, err)
		}

		c := record[0]
		if _, ok := cm[c]; !ok {
			cm[c] = record[1]

			cs = append(cs, c)
		}
	}

	cs = append(cs, "ZZ")

	cm["ZZ"] = "No country"

	return cs, cm, nil
}

func readMarkets(filename string) ([]*market, error) {
	markets := []*market{}

	f, err := os.Open(filename)
	if err != nil {
		return markets, fmt.Errorf("opening markets: %w", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Comma = ','
	r.Comment = '#'
	r.FieldsPerRecord = 13
	r.ReuseRecord = false
	r.TrimLeadingSpace = true

	ln := 0

	for {
		record, err := r.Read()
		if errors.Is(err, io.EOF) {
			break
		}

		ln++
		if ln == 1 {
			continue
		}

		if err != nil {
			return markets, fmt.Errorf("'%v' line %v: error reading file: %w", filename, ln, err)
		}

		m, err := parseMarket(record, ln, filename)
		if err != nil {
			return markets, err
		}

		markets = append(markets, m)
	}

	return markets, nil
}

//nolint:gomnd,funlen,goerr113,cyclop
func parseMarket(record []string, ln int, filename string) (*market, error) {
	country := strings.Trim(record[0], "\"")
	if len(country) < 2 {
		return nil, fmt.Errorf("'%v' line %v: country should have at least 2 characters, got '%v'", filename, ln, country)
	}

	code := strings.Trim(record[1], "\"")
	if len(code) != 2 {
		return nil, fmt.Errorf("'%v' line %v: ISO 3166 country code should have 2 characters, got '%v'", filename, ln, code)
	}

	mic := strings.Trim(record[2], "\"")
	if len(mic) != 4 {
		return nil, fmt.Errorf("'%v' line %v: MIC should have 4 characters, got '%v'", filename, ln, mic)
	}

	micOp := strings.Trim(record[3], "\"")
	if len(micOp) != 4 {
		return nil, fmt.Errorf("'%v' line %v: operational MIC should have 4 characters, got '%v'", filename, ln, micOp)
	}

	os := strings.Trim(record[4], "\"")
	if os != "O" && os != "S" {
		return nil, fmt.Errorf("'%v' line %v: o/s should be either 'O' or 'S', got '%v'", filename, ln, os)
	}

	if os == "O" && mic != micOp {
		return nil, fmt.Errorf(
			"'%v' line %v: operational MIC should have both MICs '%v', got '%v' and '%v'",
			filename, ln, micOp, mic, micOp)
	}

	if os == "S" && mic == micOp {
		return nil, fmt.Errorf(
			"'%v' line %v: segment MIC should have different O/S MICs, got equal '%v' MICs", filename, ln, mic)
	}

	name := strings.Trim(record[5], "\"")
	acronym := strings.Trim(record[6], "\"")
	city := strings.Trim(record[7], "\"")
	website := strings.ToLower(strings.Trim(record[8], "\""))

	statusDate := strings.Trim(record[9], "\"")
	if len(statusDate) < 1 {
		return nil, fmt.Errorf("'%v' line %v: status date should not be empty", filename, ln)
	}

	status := strings.Trim(record[10], "\"")
	if len(status) < 1 {
		return nil, fmt.Errorf("'%v' line %v: status should not be empty", filename, ln)
	}

	creationDate := strings.Trim(record[11], "\"")
	if len(creationDate) < 1 {
		return nil, fmt.Errorf("'%v' line %v: creation date should not be empty", filename, ln)
	}

	comments := ""
	if len(record) > 12 {
		comments = strings.Trim(record[12], "\"")
	}

	return &market{
		country, "", code, mic, micOp, os == "O", name, acronym, city, website,
		statusDate, status, creationDate, comments, 0,
	}, nil
}

//nolint:gomnd,funlen
func enrichMarkets(ms []*market, cm map[string]string) error {
	// seconds east of UTC
	tzm := map[string]int{
		"":                          0,              // Unknown (GMT)
		"AABENRAA":                  3600,           // Denmark (GMT+1)
		"AALBORG":                   3600,           // Denmark (GMT+1)
		"ABIDJAN":                   0,              // Côte d'Ivoire (GMT)
		"ABU DHABI":                 3600 * 4,       // United Arab Emirates (GMT+4)
		"ACCRA":                     0,              // Ghana (GMT)
		"AHMEDABAD":                 3600*5 + 1800,  // Gujarat, India (GMT+5:30)
		"ALGIERS":                   3600,           // Algeria (GMT+1)
		"ALMA-ATA":                  3600 * 6,       // Kazakhstan (GMT+6)
		"ALMATY":                    3600 * 6,       // Kazakhstan (GMT+6)
		"AMMAN":                     3600 * 2,       // Jordan (GMT+2)
		"AMSTERDAM":                 3600,           // The Netherlands (GMT+1)
		"ANKARA":                    3600 * 3,       // Turkey (GMT+3)
		"ANTANANARIVO":              3600 * 3,       // Madagascar (GMT+3)
		"ASTANA":                    3600 * 6,       // Nur-Sultan, Kazakhstan (GMT+6)
		"ASTI":                      3600,           // Italy (GMT+1)
		"ASUNCION":                  3600 * -4,      // Paraguay (GMT-4)
		"ATHENS":                    3600 * 2,       // Greece (GMT+2)
		"ATLANTA":                   3600 * -5,      // Atlanta, GA, USA (GMT-5)
		"AUCKLAND":                  3600 * 12,      // New Zealand (GMT+12)
		"AYLESBURY":                 0,              // UK (GMT)
		"BAGHDAD":                   3600 * 3,       // Iraq (GMT+3)
		"BAKU":                      3600 * 4,       // Azerbaijan (GMT+4)
		"BANGALORE":                 3600*5 + 1800,  // Karnataka, India (GMT+5:30)
		"BANGKOK":                   3600 * 7,       // Thailand (GMT+7)
		"BANJA LUKA":                3600,           // Bosnia and Herzegovina (GMT+1)
		"BARCELONA":                 3600,           // Spain (GMT+1)
		"BASSETERRE":                3600 * -4,      // Saint Kitts and Nevis (GMT-4)
		"BEDMINSTER":                3600 * -5,      // NJ, USA (GMT-5)
		"BEIJING":                   3600 * 8,       // China (GMT+8)
		"BEIRUT":                    3600 * 2,       // Lebanon (GMT+2)
		"BELGRADE":                  3600,           // Serbia (GMT+1)
		"BERGEN":                    3600,           // Norway (GMT+1)
		"BERLIN":                    3600,           // Germany (GMT+1)
		"BERMUDA":                   3600 * -4,      // Bermuda (GMT-4)
		"BERN":                      3600,           // Switzerland (GMT+1)
		"BERNE":                     3600,           // Switzerland (GMT+1)
		"BIELLA":                    3600,           // Province of Biella, Italy (GMT+1)
		"BILBAO":                    3600,           // Spain (GMT+1)
		"BISHKEK":                   3600 * 6,       // Kyrgyzstan (GMT+6)
		"BLANTYRE":                  3600 * 2,       // Malawi (GMT+2)
		"BOCA RATON":                3600 * -5,      // FL, USA (GMT-4)
		"BOGOTA":                    3600 * -5,      // Colombia (GMT-5)
		"BOLOGNA":                   3600,           // Italy (GMT+1)
		"BOSTON":                    3600 * -5,      // MA, USA (GMT-5)
		"BRATISLAVA":                3600,           // Slovakia (GMT+1)
		"BRIDGETOWN":                3600 * -5,      // Barbados (GMT-4)
		"BRUSSELS":                  3600,           // Belgium (GMT+1)
		"BRYANSTON, SANDTON":        3600 * 2,       // Sandton, South Africa (GMT+2)
		"BUCHAREST":                 3600 * 2,       // Romania (GMT+2)
		"BUDAPEST":                  3600,           // Hungary (GMT+1)
		"BUENOS AIRES":              3600 * -5,      // Argentina (GMT-3)
		"CAIRO":                     3600 * 2,       // Egypt (GMT+2)
		"CALCUTTA":                  3600*5 + 1800,  // West Bengal, India (GMT+5:30)
		"CALGARY":                   3600 * -7,      // AB, Canada (GMT-7)
		"CARACAS":                   3600 * -4,      // Capital District, Venezuela (GMT-4)
		"CASABLANCA":                0,              // Morocco (GMT)
		"CHARLOTTE":                 3600 * -5,      // NC, USA (GMT-5)
		"CHATHAM":                   3600*12 + 2700, // Chatham Islands Territory, New Zealand (GMT+12:45)
		"CHICAGO":                   3600 * -6,      // IL, USA (GMT-6)
		"CHISINAU":                  3600 * 2,       // Moldova (GMT+2)
		"CHITTAGONG":                3600 * 6,       // Bangladesh (GMT+6)
		"CHIYODA-KU":                3600 * 9,       // Tokyo, Japan (GMT+9)
		"CLUJ NAPOCA":               3600 * 2,       // Romania (GMT+2)
		"COLOMBO":                   3600*5 + 1800,  // Sri Lanka (GMT+5:30)
		"COPENHAGEN":                3600,           // Denmark (GMT+1)
		"CORDOBA":                   3600 * -3,      // Argentina (GMT-3)
		"CORRIENTES":                3600 * -3,      // Argentina (GMT-3)
		"CYBERCITY, EBENE":          3600 * 4,       // Mauritius (GMT+4)
		"DALIAN":                    3600 * 8,       // Liaoning, China (GMT+8)
		"DAMASCUS":                  3600 * 2,       // Syria (GMT+2)
		"DAR ES SALAAM":             3600 * 3,       // Tanzania (GMT+3)
		"DELHI":                     3600*5 + 1800,  // India (GMT+5:30)
		"DHAKA":                     3600 * 6,       // Bangladesh (GMT+6)
		"DNIPROPETROVSK":            3600 * 2,       // Dnipropetrovsk Oblast, Ukraine (GMT+2)
		"DOHA":                      3600 * 3,       // Qatar (GMT+3)
		"DOUALA":                    3600,           // Cameroon (GMT+1)
		"DUBAI":                     3600 * 4,       // United Arab Emirates (GMT+4)
		"DUBLIN":                    0,              // County Dublin, Ireland (GMT)
		"DUESSELDORF":               3600,           // Germany (GMT+1)
		"EBENE":                     3600 * 4,       // Mauritius (GMT+4)
		"EBENE CITY":                3600 * 4,       // Mauritius (GMT+4)
		"EDEN ISLAND":               3600 * 4,       // Seychelles (GMT+4)
		"EDINBURGH":                 0,              // UK (GMT)
		"EL SALVADOR":               3600 * -6,      // El Salvador (GMT-6)
		"ESCH-SUR-ALZETTE":          3600,           // Luxembourg (GMT+1)
		"ESPIRITO SANTO":            3600 * -3,      // Brazil (GMT-3)
		"ESPOO":                     3600 * 2,       // Finland (GMT+2)
		"FIAC":                      3600,           // France (GMT+1)
		"FIRENZE":                   3600,           // Italy (GMT+1)
		"FLORENCE":                  3600,           // Italy (GMT+1)
		"FRANKFURT":                 3600,           // Germany (GMT+1)
		"FRANKFURT AM MAIN":         3600,           // Germany (GMT+1)
		"FUKUOKA":                   3600 * 9,       // Japan (GMT+9)
		"GABORONE":                  3600 * 2,       // Botswana (GMT+2)
		"GANDHINAGAR":               3600*5 + 1800,  // Gujarat, India (GMT+5:30)
		"GEORGETOWN":                3600 * -4,      // Guyana (GMT-4)
		"GIBRALTAR":                 3600,           // Gibraltar (GMT+1)
		"GIFT CITY, GANDHINAGAR":    3600*5 + 1800,  // Gujarat, India (GMT+5:30)
		"GLENVIEW":                  3600 * -6,      // IL, USA (GMT-6)
		"GREAT NECK":                3600 * -5,      // NY, USA (GMT-5)
		"GREENWICH":                 0,              // UK (GMT)
		"GRINDSTED":                 3600,           // Denmark (GMT+1)
		"GUATEMALA":                 3600 * -6,      // Guatemala (GMT-6)
		"GUAYAQUIL":                 3600 * -5,      // Ecuador (GMT-5)
		"GUAYNABO":                  3600 * -4,      // Puerto Rico (GMT-4)
		"GUILDFORD":                 0,              // UK (GMT)
		"HAMBURG":                   3600,           // Germany (GMT+1)
		"HAMILTON":                  3600 * -5,      // ON, Canada (GMT-5)
		"HANNOVER":                  3600,           // Germany (GMT+1)
		"HANOI":                     3600 * 7,       // Hoàn Kiếm, Hanoi, Vietnam (GMT+7)
		"HARARE":                    3600 * 2,       // Zimbabwe (GMT+2)
		"HELSINKI":                  3600 * 2,       // Finland (GMT+2)
		"HO CHI MINH CITY":          3600 * 7,       // Vietnam (GMT+7)
		"HONG KONG":                 3600 * 7,       // Hong Kong (GMT+8)
		"HORSENS":                   3600,           // Denmark (GMT+1)
		"HOVE":                      0,              // UK (GMT)
		"INDORE MADHYA PRADESH":     3600*5 + 1800,  // Madhya Pradesh, India (GMT+5:30)
		"ISTANBUL":                  3600 * 3,       // Turkey (GMT+3)
		"JAKARTA":                   3600 * 7,       // Indonesia (GMT+7)
		"JERSEY CITY":               3600 * -5,      // NJ, USA (GMT-5)
		"JOHANNESBURG":              3600 * 2,       // South Africa (GMT+2)
		"KAMPALA":                   3600 * 3,       // Uganda (GMT+3)
		"KANSAS CITY":               3600 * -6,      // MO, USA (GMT-6)
		"KARACHI":                   3600 * 5,       // Pakistan (GMT+5)
		"KATHMANDU":                 3600*5 + 2700,  // Nepal (GMT+5:45)
		"KHARKOV":                   3600 * 2,       // Kharkiv Oblast, Ukraine (GMT+2)
		"KHARTOUM":                  3600 * 2,       // Sudan (GMT+2)
		"KIEL":                      3600,           // Germany (GMT+1)
		"KIEV":                      3600 * 2,       // Ukraine (GMT+2)
		"KIGALI":                    3600 * 2,       // Rwanda (GMT+2)
		"KINGSTON":                  3600 * -5,      // ON, Canada (GMT-5)
		"KINGSTOWN":                 3600 * -4,      // Saint Vincent and the Grenadines (GMT-4)
		"KLAGENFURT AM WOERTHERSEE": 3600,           // Austria (GMT+1)
		"KONGSVINGER":               3600,           // Norway (GMT+1)
		"KUALA LUMPUR":              3600 * 8,       // Federal Territory of Kuala Lumpur, Malaysia (GMT+8)
		"KUWAIT":                    3600 * 3,       // Kuwait (GMT+3)
		"KYIV":                      3600 * 2,       // Ukraine (GMT+2)
		"LA PAZ":                    3600 * -4,      // Bolivia (GMT-4)
		"LABUAN":                    3600 * 8,       // Labuan Federal Territory, Malaysia (GMT+8)
		"LAGOS":                     3600,           // Nigeria (GMT+1)
		"LANE COVE":                 3600 * 10,      // NSW, Australia (GMT+10)
		"LAO":                       3600 * 7,       // Laos (GMT+7)
		"LARNACA":                   3600 * 2,       // Cyprus (GMT+2)
		"LEIPZIG":                   3600,           // Germany (GMT+1)
		"LIMA":                      3600 * -5,      // Peru (GMT-5)
		"LIMASSOL":                  3600 * 2,       // Cyprus (GMT+2)
		"LINZ":                      3600,           // Austria (GMT+1)
		"LISBOA":                    0,              // Portugal (GMT)
		"LISBON":                    0,              // Portugal (GMT)
		"LJUBLJANA":                 3600,           // Slovenia (GMT+1)
		"LONDON":                    0,              // UK (GMT)
		"LUANDA":                    3600,           // Angola (GMT+1)
		"LUSAKA":                    3600 * 2,       // Zambia (GMT+2)
		"LUXEMBOURG":                3600,           // Luxembourg (GMT+1)
		"LUZERN":                    3600,           // Switzerland (GMT+1)
		"MADRAS":                    3600*5 + 1800,  // Tamil Nadu, India (GMT+5:30)
		"MADRID":                    3600,           // Spain (GMT+1)
		"MAKATI CITY":               3600 * 8,       // Metro Manila, Philippines (GMT+8)
		"MALE":                      3600 * 5,       // Maldives (GMT+5)
		"MANAGUA":                   3600 * -6,      // Nicaragua (GMT-6)
		"MANAMA":                    3600 * 3,       // Bahrain (GMT+3)
		"MANILA":                    3600 * 8,       // Metro Manila, Philippines (GMT+8)
		"MAPUTO":                    3600 * 2,       // Mozambique (GMT+2)
		"MARINGA":                   3600 * -3,      // Maringá - State of Paraná, Brazil (GMT-3)
		"MBABANE":                   3600 * 2,       // Eswatini (GMT+2)
		"MELBOURNE":                 3600 * 10,      // Melbourne VIC, Australia (GMT+10)
		"MENDOZA":                   3600 * -3,      // Mendoza Province, Argentina (GMT-3)
		"MEXICO":                    3600 * -6,      // CDMX, Mexico (GMT-6)
		"MILAN":                     3600,           // Italy (GMT+1)
		"MILANO":                    3600,           // Italy (GMT+1)
		"MINNEAPOLIS":               3600 * -6,      // MN, USA (GMT-6)
		"MINSK":                     3600 * 3,       // Belarus (GMT+3)
		"MONTENEGRO":                3600,           // Montenegro (GMT+1)
		"MONTEVIDEO":                3600 * -3,      // Montevideo Department, Uruguay (GMT-3)
		"MONTREAL":                  3600 * -5,      // QC, Canada (GMT-5)
		"MOORPARK":                  3600 * -8,      // CA, USA (GMT-8)
		"MOSCOW":                    3600 * 3,       // Russia (GMT+3)
		"MOUNT PLEASANT":            3600 * -5,      // MI, USA (GMT-5)
		"MUENCHEN":                  3600,           // Germany (GMT+1)
		"MUMBAI":                    3600*5 + 1800,  // Maharashtra, India (GMT+5:30)
		"MUNICH":                    3600,           // Germany (GMT+1)
		"MUSCAT":                    3600 * 4,       // Oman (GMT+4)
		"NABLUS":                    3600 * 2,       // Nablus (GMT+2)
		"NACKA":                     3600,           // Sweden (GMT+1)
		"NAGOYA":                    3600 * 9,       // Aichi, Japan (GMT+9)
		"NAIROBI":                   3600 * 3,       // Kenya (GMT+3)
		"NARBERTH":                  3600 * -5,      // PA, USA (GMT-5)
		"NASAU":                     3600 * -5,      // The Bahamas (GMT-5)
		"NEW JERSEY":                3600 * -5,      // USA (GMT-5)
		"NEW YORK":                  3600 * -5,      // NY, USA (GMT-5)
		"NEWCASTLE":                 0,              // UK (GMT)
		"NICOSIA":                   3600 * 2,       // Cyprus (GMT+2)
		"NICOSIA (LEFKOSIA)":        3600 * 2,       // Cyprus (GMT+2)
		"NORTH BERGEN":              3600 * -5,      // NJ, USA (GMT-5)
		"NOVOSIBIRSK":               3600 * 7,       // Novosibirsk Oblast, Russia (GMT+7)
		"NYON":                      3600,           // Switzerland (GMT+1)
		"ODESSA":                    3600 * 2,       // Odessa Oblast, Ukraine (GMT+2)
		"OLDENBURG":                 3600,           // Germany (GMT+1)
		"OSAKA":                     3600 * 9,       // Japan (GMT+9)
		"OSLO":                      3600,           // Norway (GMT+1)
		"PADOVA":                    3600,           // Province of Padua, Italy (GMT+1)
		"PALMA DE MALLORCA":         3600,           // Spain (GMT+1)
		"PANAMA":                    3600 * -5,      // Panama (GMT-5)
		"PARIS":                     3600,           // France (GMT+1)
		"PASIG CITY":                3600 * 8,       // Metro Manila, Philippines (GMT+8)
		"PHILADELPHIA":              3600 * -5,      // PA, USA (GMT-5)
		"PHNOM PENH":                3600 * 7,       // Cambodia (GMT+7)
		"PORT LOUIS":                3600 * 4,       // Mauritius (GMT+4)
		"PORT MORESBY":              3600 * 10,      // Papua New Guinea (GMT+10)
		"PORT OF SPAIN":             3600 * -4,      // Trinidad and Tobago (GMT-4)
		"PORT VILA":                 3600 * 11,      // Vanuatu (GMT+11)
		"PRAGUE":                    3600,           // Czechia (GMT+1)
		"PRAIA":                     3600 * -1,      // Cape Verde (GMT-1)
		"PRINCETON":                 3600 * -5,      // NJ, USA (GMT-5)
		"PURCHASE":                  3600 * -5,      // Harrison, NY, USA (GMT-5)
		"QUITO":                     3600 * -5,      // Ecuador (GMT-5)
		"RANDERS":                   3600,           // Denmark (GMT+1)
		"RED BANK":                  3600 * -5,      // NJ, USA (GMT-5)
		"REGENSBURG":                3600,           // Germany (GMT+1)
		"REGGIO EMILIA":             3600,           // Province of Reggio Emilia, Italy (GMT+1)
		"REYKJAVIK":                 0,              // Iceland (GMT)
		"RIGA":                      3600 * 2,       // Rīgas pilsēta, Latvia (GMT+2)
		"RIO DE JANEIRO":            3600 * -3,      // State of Rio de Janeiro, Brazil (GMT-3)
		"RIYADH":                    3600 * 3,       // Saudi Arabia (GMT+3)
		"RODGAU":                    3600,           // Germany (GMT+1)
		"ROMA":                      3600,           // Italy (GMT+1)
		"ROME":                      3600,           // Italy (GMT+1)
		"ROSARIO":                   3600 * -3,      // Santa Fe Province, Argentina (GMT-3)
		"S-HERTOGENBOSCH":           3600,           // The Netherlands (GMT+1)
		"SABADELL":                  3600,           // Spain (GMT+1)
		"SAINT-PETERSBURG":          3600 * 3,       // Russia (GMT+3)
		"SALZBURG":                  3600,           // Austria (GMT+1)
		"SAMARA":                    3600 * 4,       // Samara Oblast, Russia (GMT+4)
		"SAN CARLOS":                3600 * -8,      // CA, USA (GMT-8)
		"SAN FRANCISCO":             3600 * -8,      // CA, USA (GMT-8)
		"SAN JOSE":                  3600 * -8,      // CA, USA (GMT-8)
		"SANTA FE":                  3600 * -7,      // NM, USA (GMT-7)
		"SANTANDER":                 3600,           // Spain (GMT+1)
		"SANTIAGO":                  3600 * -4,      // Chile (GMT-4)
		"SANTO DOMINGO":             3600 * -4,      // Dominican Republic (GMT-4)
		"SAO PAULO":                 3600 * -3,      // State of São Paulo, Brazil (GMT-3)
		"SAPPORO":                   3600 * 9,       // Hokkaido, Japan (GMT+9)
		"SARAJEVO":                  3600,           // Bosnia and Herzegovina (GMT+1)
		"SEOUL":                     3600 * 9,       // South Korea (GMT+9)
		"SHANGHAI":                  3600 * 8,       // China (GMT+8)
		"SHENZHEN":                  3600 * 8,       // Guangdong Province, China (GMT+8)
		"SILKEBORG":                 3600,           // Denmark (GMT+1)
		"SINGAPORE":                 3600 * 8,       // Singapore (GMT+8)
		"SKOPJE":                    3600,           // North Macedonia (GMT+1)
		"SLIEMA":                    3600,           // Malta (GMT+1)
		"SOFIA":                     3600 * 2,       // Bulgaria (GMT+2)
		"SPLIT":                     3600,           // Croatia (GMT+1)
		"ST ALBANS":                 0,              // UK (GMT)
		"ST.  PETER PORT":           0,              // Guernsey (GMT)
		"STAMFORD":                  3600 * -5,      // CT, USA (GMT-5)
		"STOCKHOLM":                 3600,           // Sweden (GMT+1)
		"STUTTGART":                 3600,           // Germany (GMT+1)
		"SUVA":                      3600 * 12,      // Fiji (GMT+12)
		"SYDNEY":                    3600 * 10,      // NSW, Australia (GMT+10)
		"TAIPEI":                    3600 * 8,       // Taiwan (GMT+8)
		"TAIWAN":                    3600 * 8,       // Taiwan (GMT+8)
		"TALLINN":                   3600 * 2,       // Harju County, Estonia (GMT+2)
		"TASHKENT":                  3600 * 5,       // Uzbekistan (GMT+5)
		"TBILISI":                   3600 * 4,       // Georgia (GMT+4)
		"TEGUCIGALPA":               3600 * -6,      // Honduras (GMT-6)
		"TEHRAN":                    3600*3 + 1800,  // Iran (GMT+3:30)
		"TEL AVIV":                  3600 * 2,       // Yafo, Israel (GMT+2)
		"THE HAGUE":                 3600,           // The Netjerlands (GMT+1)
		"TIRANA":                    3600,           // Albania (GMT+1)
		"TOKYO":                     3600 * 9,       // Japan (GMT+9)
		"TORINO":                    3600,           // Italy (GMT+1)
		"TORONTO":                   3600 * -5,      // ON, Canada (GMT-5)
		"TORSHAVN":                  0,              // Faroe Islands (GMT)
		"TRIPOLI":                   3600 * 2,       // Libya (GMT+2)
		"TROMSO":                    3600,           // Norway (GMT+1)
		"TRONDHEIM":                 3600,           // Norway (GMT+1)
		"TUCUMAN":                   3600 * -3,      // Tucumán, Argentina (GMT-3)
		"TUNIS":                     3600,           // Tunisia (GMT+1)
		"ULAAN BAATAR":              3600 * 8,       // Mongolia (GMT+8)
		"UNTERSCHLEISSHEM":          3600,           // Germany (GMT+1)
		"UTRECHT":                   3600,           // The Netherlands (GMT+1)
		"VADUZ":                     3600,           // Liechtenstein (GMT+1)
		"VALENCIA":                  3600,           // Spain (GMT+1)
		"VALLETTA":                  3600,           // Malta (GMT+1)
		"VALPARAISO":                3600 * -6,      // IN, USA (GMT-6)
		"VANCOUVER":                 3600 * -8,      // BC, Canada (GMT-8)
		"VICTORIA":                  3600 * 10,      // Australia (GMT+10)
		"VIENNA":                    3600,           // Austria (GMT+1)
		"VILA":                      3600 * 11,      // Vanuatu (GMT+11)
		"VILNIUS":                   3600 * 2,       // Lithuania (GMT+2)
		"WARSAW":                    3600,           // Poland (GMT+1)
		"WARSZAWA":                  3600,           // Poland (GMT+1)
		"WASHINGTON":                3600 * -8,      // USA (GMT-8)
		"WASHINGTON/NEW YORK":       3600 * -5,      // NY, USA (GMT-5)
		"WELLINGTON":                3600 * 12,      // New Zealand (GMT+12)
		"WILLEMSTAD":                3600 * -4,      // Curaçao (GMT-4)
		"WILMINGTON":                3600 * -5,      // DE, USA (GMT-5)
		"WINDHOEK":                  3600 * 2,       // Namibia (GMT+2)
		"WINNIPEG":                  3600 * -6,      // MB, Canada (GMT-6)
		"WROCLAW":                   3600,           // Poland (GMT+1)
		"WUXI":                      3600 * 8,       // Jiangsu, China (GMT+8)
		"YEREVAN":                   3600 * 4,       // Armenia (GMT+4)
		"ZAGREB":                    3600,           // Croatia (GMT+1)
		"ZARAGOZA":                  3600,           // Spain (GMT+1)
		"ZHENGZHOU":                 3600 * 8,       // Henan, China (GMT+8)
		"ZILINA":                    3600,           // Slovakia (GMT+1)
		"ZURICH":                    3600,           // Switzerland (GMT+1)
		"ZZ":                        0,              // Unknown (GMT)
	}

	for _, m := range ms {
		if v, ok := cm[m.code]; ok {
			m.countryInc = v
		}

		t, ok := tzm[m.city]
		if !ok {
			return fmt.Errorf("'%v': %w", m.city, errUnknownTimezoneCity)
		}

		m.tzsec = t
	}

	return nil
}

func arrangeByCountry(ms []*market, cs []string) []*mic {
	ams := []*mic{}

	// colllect operational MICs odered by country
	for _, c := range cs {
		for _, m := range ms {
			if m.code == c && m.isOperational {
				// enrich operational MIC with segment MICs if any
				ss := []*market{}

				for _, s := range ms {
					if s.micOp == m.micOp && !s.isOperational {
						ss = append(ss, s)
					}
				}

				ams = append(ams, &mic{m, ss})
			}
		}
	}

	return ams
}

func collectTimeZones(ms []*mic) (map[int][]*market, []int) {
	tzm := map[int][]*market{}
	tzs := []int{}

	for _, m := range ms {
		if v, ok := tzm[m.oper.tzsec]; ok {
			tzm[m.oper.tzsec] = append(v, m.oper)
		} else {
			tzm[m.oper.tzsec] = []*market{m.oper}
			tzs = append(tzs, m.oper.tzsec)
		}

		for _, s := range m.segs {
			if v, ok := tzm[s.tzsec]; ok {
				tzm[s.tzsec] = append(v, s)
			} else {
				tzm[s.tzsec] = []*market{s}
				tzs = append(tzs, s.tzsec)
			}
		}
	}

	return tzm, tzs
}

func arrangeByTimeZone(ms []*mic) []*tzmic {
	tzm, tzs := collectTimeZones(ms)
	sort.Ints(tzs)

	tzmics := make([]*tzmic, 0, len(tzs))
	for _, t := range tzs {
		tzmics = append(tzmics, &tzmic{t, tzm[t]})
	}

	return tzmics
}

func collectExcludedCountries(ms []*market, icm map[string]string) []string {
	type void struct{}

	var v void

	ecm := map[string]void{}
	ecs := []string{}

	for _, m := range ms {
		if _, ok := icm[m.code]; !ok {
			if _, ok := ecm[m.code]; !ok {
				ecm[m.code] = v

				ecs = append(ecs, m.code)
			}
		}
	}

	sort.Strings(ecs)

	return ecs
}

func collectIncludedCountries(ms []*market, icm map[string]string) []string {
	type void struct{}

	var v void

	ncm := map[string]void{}
	ncs := []string{}

	for _, m := range ms {
		if _, ok := icm[m.code]; ok {
			if _, ok := ncm[m.code]; !ok {
				ncm[m.code] = v

				ncs = append(ncs, m.code)
			}
		}
	}

	sort.Strings(ncs)

	return ncs
}

func die(err error) {
	fmt.Println(err) //nolint:forbidigo
	os.Exit(1)
}

func printf(w io.Writer, format string, a ...interface{}) {
	_, err := fmt.Fprintf(w, format, a...)
	if err != nil {
		die(err)
	}
}

func concatenateCountries(ss []string) string {
	if len(ss) == 0 {
		return "none"
	}

	var b strings.Builder

	first := true
	for _, s := range ss {
		if first {
			first = false
		} else {
			b.WriteString(", ")
		}

		b.WriteString(s)
	}

	return b.String()
}

func safeMic(s string) string {
	// some MICs begin with digits, prepend them with an X
	if unicode.IsDigit(rune(s[0])) {
		s = "X" + s
	}

	return s
}

func concatenateMics(ms []*market) string {
	var b strings.Builder

	first := true
	for _, m := range ms {
		if first {
			first = false
		} else {
			b.WriteString(", ")
		}

		b.WriteString(safeMic(m.mic))
	}

	return b.String()
}

//nolint:misspell,cyclop
func printMicPython(w io.Writer, m *market) {
	printf(w, "    %v = MIC('%v', '%v', '%v', %v)\n", safeMic(m.mic), m.mic, m.micOp, m.code, m.tzsec)

	if m.isOperational {
		printf(w, "    \"\"\"%v - operational", safeMic(m.mic))
	} else {
		printf(w, "    \"\"\"%v - segment of %v", safeMic(m.mic), m.micOp)
	}

	if m.name != "" {
		printf(w, ": %v", m.name)

		if !strings.HasSuffix(m.name, ".") {
			printf(w, ".")
		}
	}

	// Do not show comments if the text is already present in the name.
	if m.comments != "" &&
		!strings.Contains(strings.ToUpper(m.name), strings.ToUpper(m.comments)) &&
		!strings.Contains(strings.ToUpper(m.name), strings.ToUpper(strings.Trim(m.comments, "."))) {
		printf(w, " %v", m.comments)

		if !strings.HasSuffix(m.comments, ".") {
			printf(w, ".")
		}
	}

	if m.city != "" && m.city != "ZZ" {
		s := strings.Title(strings.ToLower(m.city))

		printf(w, " Location: %v (%v), %v", m.countryInc, m.code, s)

		if m.website != "" {
			printf(w, ", %v", m.website)

			if !strings.HasSuffix(m.website, ".") {
				printf(w, ".")
			}
		} else if !strings.HasSuffix(s, ".") {
			printf(w, ".")
		}
	}

	printf(w, "\"\"\"\n")
}

func printBuffer(b *bytes.Buffer, filename string, formatSource bool) {
	f, err := os.Create(filename)
	if err != nil {
		die(err)
	}
	defer f.Close()

	if formatSource {
		data, err := format.Source(b.Bytes())
		if err != nil {
			die(err)
		}

		_, err = f.Write(data)
		if err != nil {
			die(err)
		}
	} else {
		_, err = f.Write(b.Bytes())
		if err != nil {
			die(err)
		}
	}
}

//nolint:funlen
func printMicsPython(filename string, ms []*mic, ics []string, ecs []string, tzmics []*tzmic) {
	var b bytes.Buffer

	printf(&b, "# Code generated by 'go generate'; DO NOT EDIT.\n")
	printf(&b, "# %v\n\n", time.Now())

	printf(&b, "# Data source: https://www.iso20022.org/sites/default/files/ISO10383_MIC/ISO10383_MIC.csv\n")
	printf(&b, "# Data source publication date: %v\n\n", dataPublicationDate)

	printf(&b, "# Included: MICs for countries with ISO 3166 alpha-2 codes\n")
	printf(&b, "# %v\n", concatenateCountries(ics))
	printf(&b, "# Excluded: MICs for countries with ISO 3166 alpha-2 codes\n")
	printf(&b, "# %v\n\n", concatenateCountries(ecs))

	printf(&b, "# Package mics provides a set of predefined ISO 10383 Market Identifier Codes\n")
	printf(&b, "# generated from data from https://www.iso20022.org/market-identifier-codes\n")
	printf(&b, "# with data publication date %v.\n", dataPublicationDate)
	printf(&b, "#\n")
	printf(&b, "# MICs are generated for countries with the following ISO 3166 alpha-2 codes:\n")
	printf(&b, "# %v.\n", concatenateCountries(ics))

	printf(&b, "class MICs(metaclass=SubscriptableType):\n\n")

	for _, m := range ms {
		printMicPython(&b, m.oper)

		for _, s := range m.segs {
			printMicPython(&b, s)
		}
	}

	printBuffer(&b, filename, false)
}
