# the 50 states (the FRED series names are e.g. 'ALRGSP' and 'ALPOP' for Alabama)
STATES = [
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA',
    'HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
    'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC',
    'SD','TN','TX','UT','VT','VA','WA','WV','WI','WY',
]

# the four Census regions
_REGION_STATES = {
    'Northeast': 'CT ME MA NH RI VT NJ NY PA',
    'Midwest': 'IL IN MI OH WI IA KS MN MO NE ND SD',
    'South': 'DE FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX',
    'West': 'AZ CO ID MT NV NM UT WY AK CA HI OR WA',
}

# state -> region, e.g. REGION['MS'] == 'South'
REGION = {state: region for region, states in _REGION_STATES.items() for state in states.split()}

assert sorted(REGION.keys()) == sorted(STATES), 'the region mapping does not cover all 50 states'
