This project is a Flask-based Pokémon Pokedex application that enables users to filter Pokémon by attributes such as Type, Generation, and Legendary status, as well as add new entries to the dataset. It uses two machine-learning models—a Linear Regression model to predict the “Total” stat of a Pokémon and a Random Forest classifier to predict its Legendary status. The front-end allows users to search by various criteria and provides data visualizations in the form of charts and graphs, offering a clear representation of the Pokémon dataset. When new Pokémon are added, the system predicts and displays their total stats and Legendary status. Users can then choose to save these new entries, which are appended to the CSV dataset and made persistent.

The application code is divided into multiple routes. The home route loads the interface that presents various filtering options. Upon submission of these filters, the back-end processes the criteria, filters the DataFrame, and returns the results. These results include a search history feature so users can revisit previous queries. When a user opts to add a new Pokémon, the application leverages the previously trained models to predict total stats and Legendary status before allowing the user to finalize and save the entry. For visualization, a set of charts are automatically generated to display Type distributions (Type 1 and Type 2) and Pokémon distribution by Generation. Additionally, a utility function formats some complex Pokémon names for a more user-friendly display. Overall, the project highlights how a small web application can integrate data science and machine-learning pipelines to enrich user experiences in a simple and interactive way.

Below are brief bullet points explaining key functions:

- train_models()
  Trains and saves the Linear Regression and Random Forest models upon startup, generating and saving model evaluation plots.

- clean_names(results)  
  Cleans Pokémon names (handling prefixes like "Mega" or “Primal”) and replaces missing Type 2 values with “None.”

- generate_charts(dataframe)  
  Creates pie charts for Type 1 and Type 2 distributions, plus a bar chart for Pokémon distribution by Generation, all saved in the `static` directory.

- find_pokemon()
  Filters the Pokémon DataFrame based on user input (type, generation, legendary) and returns the matching records with associated visualizations.

- search(search_id)  
  Retrieves past filter criteria from the search history and repeats the search, returning consistent results.

- view_database() / view_pokedex()  
  Renders the entire Pokémon dataset (cleaned or raw) in a simple table view.

- add_pokemon()
  Accepts form data for a new Pokémon, uses the trained models to predict its total stats and Legendary status, and displays the results.

- save_pokemon()
  Saves the newly added Pokémon (complete with predicted values) to the dataset, updating `pokemon_dataset.csv` for future reference.
