from .aspect import *

class Query:
    """
    A class represents a user's travel destinations query.
    A query contains a list of preferences and constraints.

    :param description (str): A brief description of the query.
    :param preferences (list[Preference]): A list of user preferences.
    :param constraints (list[Constraint]): A list of conditions that the query must satisfy.
    """
    def __init__(self, description: str, preferences: list[Preference] = [], constraints: list[Constraint] = [], hybrids: list[Hybrid] = []):
        self.description = description
        self.preferences = preferences
        self.constraints = constraints
        self.hybrids = hybrids

    def get_all_aspects(self) -> list[Aspect]:
        """
        Return a list of aspects in user query.
        """
        # otherwise, return all aspects
        return self.get_preferences() + self.get_constraints() + self.get_hybrids()
        
    def get_description(self) -> str:
        """
        Return the description of the query.
        """
        return self.description
    
    def get_preferences(self) -> list[Preference]:
        """
        Return a list of user preferences.
        """
        return self.preferences
    
    def get_constraints(self) -> list[Constraint]:
        """
        Return a list of user constraints.
        """
        return self.constraints
        
    def get_hybrids(self) -> list[Hybrid]:
        """
        Return a list of hybrids.
        """
        return self.hybrids
    
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

    def add_hybrid(self, hybrid: Hybrid):
        """
        Adds a hybrid aspect to the query.

        :param hybrid (Hybrid): The hybrid to be added.
        """
        self.hybrids.append(hybrid)