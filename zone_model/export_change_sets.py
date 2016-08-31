import pickle
#change_sets = pickle.load(open("./data/change_sets.pkl", "rb"))

year = 2011
pods = ['hlcm', 'elcm', 'rdplcm', 'repm_rent', 'repm_value']
additions = {}
updates = {}
for pod in pods:
    change_sets = pickle.load(open("./data/change_sets_%s_%s.pkl" % (year, pod), "rb"))

    for add_batch in change_sets['added'].items():
        agent_type = add_batch[0]
        agent_data = add_batch[1]
        additions[(pod, agent_type)] = agent_data

    for update_batch in change_sets['updated'].items():
        updated_table = update_batch[0][1]
        updated_column = update_batch[0][2]
        updated_data = update_batch[1]
        updates[(pod, updated_table, updated_column)] = updated_data

for pod_additions in additions.items():
    pod = pod_additions[0][0]
    agent_type = pod_additions[0][1]
    agent_data = pod_additions[1]

for pod_update in updates.items():
    pod = pod_update[0][0]
    updated_table = pod_update[0][1]
    updated_column = pod_update[0][2]
    agent_data = pod_update[1]

import pdb; pdb.set_trace()

# Stream to db
pass

#change_sets['added']['households'].to_csv('./data/hh_test.csv')
