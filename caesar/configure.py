import requests

# requires an oauth token - https://zooniverse.github.io/caesar/#configuring-caesar-via-the-api

'''
{
"if":
   ["gte",
       ["lookup", "total_classifications"],
       ["lookup", "subject.#requested_retirement_limit"]
   ]
"then":
   [{"action": "retire_subject", "reason": "requested_retirement_limit_reached"}]
}
'''

'''
{
  "extractors_config": {
    "who": {"type": "who"},
    "swap": {"type": "external", "url": "https://darryls-server.com"} # OPTIONAL
  },
  "reducers_config": {
    "swap": {"type": "external"},
    "count": {"type": "count"}
  }
  "rules_config": [
    {"if": [RULES], "then": [{"action": "retire_subject"}]}
  ]
}
'''
