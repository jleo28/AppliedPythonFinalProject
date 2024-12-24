# Joseph Leo, jaleo@usc.edu
# ITP 216, Spring 2024
# Section: 32081
# Final Project
# Description: Discover and explore your favorite pokemon. search using various criteria or add new entries to your pokedex.

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

app = Flask(__name__)

# load pokemon dataset
pokemon_df = pd.read_csv('pokemon_dataset.csv')

# global list search history
search_history = []

# global variables for models
lr_model = None
rf_model = None
X_columns = None

# colors for pokemon types
type_colors = {
    "Normal": "#A8A77A", "Fire": "#EE8130", "Water": "#6390F0", "Electric": "#F7D02C",
    "Grass": "#7AC74C", "Ice": "#96D9D6", "Fighting": "#C22E28", "Poison": "#A33EA1",
    "Ground": "#E2BF65", "Flying": "#A98FF3", "Psychic": "#F95587", "Bug": "#A6B91A",
    "Rock": "#B6A136", "Ghost": "#735797", "Dragon": "#6F35FC", "Dark": "#705746",
    "Steel": "#B7B7CE", "Fairy": "#D685AD"
}

# train models on startup
def train_models():
    global lr_model, rf_model, X_columns

    # preprocess linear regression
    lr_data = pokemon_df.dropna(subset=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total'])
    X_total = lr_data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
    y_total = lr_data['Total']

    lr_model = LinearRegression()
    lr_model.fit(X_total, y_total)

    # evaluate visualize linear regression model
    y_pred_lr = lr_model.predict(X_total)
    plt.figure(figsize=(6,6))
    plt.scatter(y_total, y_pred_lr, alpha=0.5, color='blue')
    plt.plot([y_total.min(), y_total.max()], [y_total.min(), y_total.max()], 'r--')
    plt.xlabel("actual total")
    plt.ylabel("predicted total")
    plt.title("linear regression: actual vs. predicted total")
    plt.tight_layout()
    plt.savefig('static/lr_actual_vs_predicted.png')
    plt.close()

    # preprocess random forest
    rf_data = pokemon_df.dropna(subset=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total', 'Legendary'])
    X_legendary = rf_data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']]
    y_legendary = rf_data['Legendary'].astype(int)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_legendary, y_legendary)

    X_columns = X_legendary.columns

    # feature importance visualization
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,6))
    plt.bar(X_columns[indices], importances[indices], color='green')
    plt.xlabel('features')
    plt.ylabel('importance')
    plt.title('random forest feature importances')
    plt.tight_layout()
    plt.savefig('static/rf_feature_importances.png')
    plt.close()

    return {
        'total_model': lr_model,
        'legendary_model': rf_model
    }

@app.route('/')
def home():
    # fetch unique types type1 type2
    types = sorted(pokemon_df['Type 1'].dropna().unique().tolist())
    type2 = ['None'] + types
    generations = sorted(pokemon_df['Generation'].dropna().unique().tolist())

    return render_template('home.html', types=types, type2=type2, generations=generations, search_history=search_history)

@app.route('/find_pokemon', methods=['POST'])
def find_pokemon():
    type1 = request.form.get('type1')
    type2 = request.form.get('type2')
    generation = request.form.get('generation')
    legendary = request.form.get('legendary')

    # case no input
    if not (type1 or type2 or generation or legendary):
        return render_template('results.html', error="Error: Please provide at least one input!")

    # set default legendary
    if legendary not in ["yes", "no"]:
        legendary = "no"

    if type2 == "None" or not type2:
        type2 = None

    # start filter dataframe
    filtered_df = pokemon_df
    if type1:
        filtered_df = filtered_df[filtered_df['Type 1'] == type1]
    if type2 is not None:
        filtered_df = filtered_df[filtered_df['Type 2'].fillna("") == type2]

    # handle generation filtering
    if generation and generation != "Select a Generation":
        if generation == "All":
            filtered_df = filtered_df[(filtered_df['Generation'] >= 1) & (filtered_df['Generation'] <= 6)]
        elif generation.isdigit():
            filtered_df = filtered_df[filtered_df['Generation'] == int(generation)]

    if legendary == "yes":
        filtered_df = filtered_df[filtered_df['Legendary'] == True]
    elif legendary == "no":
        filtered_df = filtered_df[filtered_df['Legendary'] == False]

    # case no results
    if filtered_df.empty:
        return render_template('results.html', no_results="No Pokémon matched your criteria.")

    # add to search history
    search_number = len(search_history) + 1
    search_history.append({
        'search_number': search_number,
        'type1': type1 or 'None',
        'type2': type2 or 'None',
        'generation': generation or 'None',
        'legendary': "Yes" if legendary == "yes" else "No"
    })

    # generate results charts
    results = clean_names(filtered_df.to_dict(orient='records'))
    generate_charts(filtered_df)
    return render_template('results.html', results=results)

@app.route('/search/<int:search_id>', methods=['GET'])
def search(search_id):
    # get search criteria history
    if 1 <= search_id <= len(search_history):
        search = search_history[search_id - 1]

        # retrieve criteria
        type1 = search['type1']
        type2 = search['type2']
        generation = search['generation']
        legendary = search['legendary']

        # re-filter dataset
        filtered_df = pokemon_df
        if type1 != 'None':
            filtered_df = filtered_df[pokemon_df['Type 1'] == type1]
        if type2 != 'None':
            filtered_df = filtered_df[pokemon_df['Type 2'].fillna("") == type2]
        if generation != 'None' and generation != "Select a Generation":
            filtered_df = filtered_df[pokemon_df['Generation'] == int(generation)]
        if legendary == "Yes":
            filtered_df = filtered_df[pokemon_df['Legendary'] == True]
        elif legendary == "No":
            filtered_df = filtered_df[pokemon_df['Legendary'] == False]

        # case no results
        if filtered_df.empty:
            return render_template('results.html', no_results="No Pokémon matched your criteria.")

        # generate graphs return results
        results = clean_names(filtered_df.to_dict(orient='records'))
        generate_charts(filtered_df)
        return render_template('results.html', results=results)
    else:
        # invalid search id
        return render_template('results.html', error="Invalid search ID!")

@app.route('/view_database', methods=['GET'])
def view_database():
    # clean pokedex names
    cleaned_pokedex = clean_names(pokemon_df.to_dict(orient='records'))

    return render_template('pokedex.html', pokedex=cleaned_pokedex)

def clean_names(results):
    prefixes = ['Mega', 'Primal']
    suffixes = ['Forme', 'Mode', 'Size']

    for r in results:
        name = r['Name']
        for prefix in prefixes:
            if prefix in name:
                idx = name.find(prefix)
                if idx > 0:
                    base_name = name[idx + len(prefix):].strip()
                    r['Name'] = f"{prefix} {base_name}"
                    break

        for suffix in suffixes:
            if suffix in name:
                suffix_idx = name.find(suffix)
                split_idx = suffix_idx
                while split_idx > 0:
                    if name[split_idx - 1].isupper() and name[split_idx].islower():
                        break
                    split_idx -= 1
                if split_idx > 0:
                    r['Name'] = name[:split_idx - 1] + ' ' + name[split_idx - 1:]
                break

        if isinstance(r['Type 2'], float) and np.isnan(r['Type 2']):
            r['Type 2'] = 'None'

    return results

def generate_charts(dataframe):
    # type1 pie chart
    type1_counts = dataframe['Type 1'].value_counts()
    type1_colors = [type_colors.get(t, "#CCCCCC") for t in type1_counts.index]

    plt.figure(figsize=(8, 8))
    plt.pie(
        type1_counts,
        labels=type1_counts.index,
        colors=type1_colors,
        autopct='%1.1f%%',
        pctdistance=0.9,
        labeldistance=1.15,
        startangle=140
    )
    plt.title("type 1 distribution")
    plt.tight_layout()
    plt.savefig('static/type1_pie.png')
    plt.close()

    # type2 pie chart
    type2_counts = dataframe['Type 2'].fillna('None').value_counts()
    type2_colors = [type_colors.get(t, "#CCCCCC") for t in type2_counts.index]

    plt.figure(figsize=(8, 8))
    plt.pie(
        type2_counts,
        labels=type2_counts.index,
        colors=type2_colors,
        autopct='%1.1f%%',
        pctdistance=0.9,
        labeldistance=1.15,
        startangle=140
    )
    plt.title("type 2 distribution")
    plt.tight_layout()
    plt.savefig('static/type2_pie.png')
    plt.close()

    # generation bar chart
    generation_counts = dataframe['Generation'].value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    plt.bar(generation_counts.index, generation_counts.values, color="#6390F0")
    plt.xlabel("generation")
    plt.ylabel("number of pokemon")
    plt.title("pokemon distribution by generation")
    plt.xticks(generation_counts.index)
    plt.savefig('static/generation_bar.png')
    plt.close()

@app.route('/add_pokemon', methods=['POST'])
def add_pokemon():
    # retrieve user input
    name = request.form.get('name')
    type1 = request.form.get('type1')
    type2 = request.form.get('type2')
    hp = int(request.form.get('hp'))
    attack = int(request.form.get('attack'))
    defense = int(request.form.get('defense'))
    sp_atk = int(request.form.get('sp_atk'))
    sp_def = int(request.form.get('sp_def'))
    speed = int(request.form.get('speed'))
    generation = int(request.form.get('generation'))

    # build initial pokemon data
    new_pokemon = {
        'Name': name,
        'Type 1': type1,
        'Type 2': type2,
        'HP': hp,
        'Attack': attack,
        'Defense': defense,
        'Sp. Atk': sp_atk,
        'Sp. Def': sp_def,
        'Speed': speed,
        'Total': None,
        'Generation': generation,
        'Legendary': None
    }

    # predict total stats
    total_input = np.array([[hp, attack, defense, sp_atk, sp_def, speed]])
    predicted_total = lr_model.predict(total_input)[0]
    new_pokemon['Total'] = int(round(predicted_total))

    # predict legendary status
    legendary_input = np.array([[hp, attack, defense, sp_atk, sp_def, speed, new_pokemon['Total']]])
    predicted_legendary = rf_model.predict(legendary_input)[0]
    new_pokemon['Legendary'] = bool(predicted_legendary)

    return render_template(
        'ml_results.html',
        pokemon=new_pokemon,
        total=new_pokemon['Total'],
        legendary="yes" if new_pokemon['Legendary'] else "no",
        save=True
    )

@app.route('/save_pokemon', methods=['POST'])
def save_pokemon():
    global pokemon_df

    # retrieve pokemon data
    new_pokemon = {
        '#': None,
        'Name': request.form.get('name'),
        'Type 1': request.form.get('type1'),
        'Type 2': request.form.get('type2'),
        'HP': int(request.form.get('hp')),
        'Attack': int(request.form.get('attack')),
        'Defense': int(request.form.get('defense')),
        'Sp. Atk': int(request.form.get('sp_atk')),
        'Sp. Def': int(request.form.get('sp_def')),
        'Speed': int(request.form.get('speed')),
        'Total': int(request.form.get('total')),
        'Generation': int(request.form.get('generation')),
        'Legendary': request.form.get('legendary') == "yes"
    }

    # determine next pokemon number
    if pokemon_df['#'].notnull().any():
        max_number = pokemon_df['#'].max()
    else:
        max_number = 721.0

    new_pokemon['#'] = max_number + 1.0

    # append to dataset
    new_row = pd.DataFrame([new_pokemon])
    pokemon_df = pd.concat([pokemon_df, new_row], ignore_index=True)

    # save updated dataset
    pokemon_df.to_csv('pokemon_dataset.csv', index=False)

    return render_template(
        'ml_results.html',
        pokemon=new_pokemon,
        total=new_pokemon['Total'],
        legendary="yes" if new_pokemon['Legendary'] else "no"
    )

@app.route('/view_pokedex', methods=['GET'])
def view_pokedex():
    # get pokedex entries
    pokedex = pokemon_df.to_dict(orient='records')
    return render_template('pokedex.html', pokedex=pokedex)

def predict_total():
    global pokemon_df

    # preprocess data
    data = pokemon_df.copy()
    data = data.dropna(subset=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total'])

    # features target
    X = data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]
    y = data['Total']

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    # evaluate model
    y_pred = model.predict(X_test)
    print("linear regression mean squared error:", mean_squared_error(y_test, y_pred))

    return model

# add total stat new pokemon
def add_total_stat_to_new_pokemon(new_pokemon):
    model = predict_total()

    # extract features
    new_features = np.array([
        [
            new_pokemon['HP'],
            new_pokemon['Attack'],
            new_pokemon['Defense'],
            new_pokemon['Sp. Atk'],
            new_pokemon['Sp. Def'],
            new_pokemon['Speed']
        ]
    ])

    # predict total stat
    predicted_total = model.predict(new_features)[0]
    new_pokemon['Total'] = round(predicted_total)

    # add to dataset
    new_row = pd.DataFrame([new_pokemon])
    pokemon_df = pd.concat([pokemon_df, new_row], ignore_index=True)

    # save updated dataset
    pokemon_df.to_csv('pokemon_dataset.csv', index=False)

    return predicted_total

def predict_legendary_rf():
    global pokemon_df

    # preprocess data
    data = pokemon_df.copy()
    data = data.dropna(subset=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total', 'Legendary'])
    data['Legendary'] = data['Legendary'].astype(int)

    # features target
    X = data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Total']]
    y = data['Legendary']

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train random forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # evaluate model
    y_pred = model.predict(X_test)
    print("random forest accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model

# add legendary status new pokemon
def add_legendary_status_to_new_pokemon(new_pokemon):
    model = predict_legendary_rf()

    # extract features
    new_features = np.array([
        [
            new_pokemon['HP'],
            new_pokemon['Attack'],
            new_pokemon['Defense'],
            new_pokemon['Sp. Atk'],
            new_pokemon['Sp. Def'],
            new_pokemon['Speed'],
            new_pokemon['Total']
        ]
    ])

    # predict legendary status
    predicted_legendary = model.predict(new_features)[0]
    new_pokemon['Legendary'] = bool(predicted_legendary)

    # add to dataset
    new_row = pd.DataFrame([new_pokemon])
    pokemon_df = pd.concat([pokemon_df, new_row], ignore_index=True)

    # save updated dataset
    pokemon_df.to_csv('pokemon_dataset.csv', index=False)

    return predicted_legendary

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    train_models()
    app.run(debug=True)
