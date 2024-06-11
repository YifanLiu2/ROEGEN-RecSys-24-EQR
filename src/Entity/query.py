from aspect import *

class Query:
    """
    A class represents a user's travel destinations query.
    A query contains a list of preferences and constraints.

    :param description (str): A brief description of the query.
    :param preferences (list[Preference]): A list of user preferences.
    :param constraints (list[Constraint]): A list of conditions that the query must satisfy.
    """
    def __init__(self, description: str, preferences: list[Preference], constraints: list[Constraint]):
        self.description = description
        self.preferences = preferences
        self.constraints = constraints
    
    def get_preferences(self) -> list[Preference]:
        """
        Return the list of user preferences.
        """
        return self.preferences
    
    def get_constraints(self) -> list[Constraint]:
        """
        Return the list of constraints.
        """
        return self.constraints
    
    def add_preference(self, preference: Preference):
        """
        Adds a preference to the query.

        :param preference (Preference): The preference to be added.
        """
        self.preferences.append(preference)
    
    def add_constraint(self, constraint: Constraint):
        """
        Adds a constraint to the query.

        :param constraint (Constraint): The constraint to be added.
        """
        self.constraints.append(constraint)
