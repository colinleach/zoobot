import json

import requests

# requires an oauth token - https://zooniverse.github.io/caesar/#configuring-caesar-via-the-api
# get this via dev tools network panel - watch for GET requests when signing in

if __name__ == '__main__':

  with open('auth_token.txt', 'r') as f:
    auth_token = f.readline()

  # retirement_rule = {
  #   "if":
  #     ["gte",
  #         ["lookup", "total_classifications"],
  #         ["lookup", "subject.#requested_retirement_limit"]
  #     ],
  #   "then":
  #     [{"action": "retire_subject", "reason": "requested_retirement_limit_reached"}]
  #   }

  # {"count": {"type": "count"}

  caesar_config = {
      "extractors": {"pluck_field": {"type": "pluck_field"}},
      "reducers": {"count": {"type": "count"}},
      "rules": []
    }

  staging_project_id = '8751'
  enhanced_workflow_id = '9814'

  response = requests.post(
    url='https://caesar.zooniverse.org/workflows/?id={}/workflows/{}'.format(
      staging_project_id,
      enhanced_workflow_id
    ),
    data=json.dumps(caesar_config),
    headers={"Authorization": "Bearer {}".format(auth_token)}
  )
  print(response)

  '''
  One example from the docs uses '_config' field names...
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
