from abc import ABC, abstractmethod

class Constraint(ABC):
    """
    An abstract base class representing a constraint with a description and verifiability.
    """
    def __init__(self, description, is_verifiable):
        self.description = description
        self.definition = None
        self.is_verifiable = is_verifiable

    def verifiable(self) -> bool:
        """
        Checks if the constraint is verifiable.
        
        :return: Boolean indicating verifiability.
        """
        return self.is_verifiable
    
    def define(self, definition: str):
        """
        Adds a definition to the constraint.
        
        :param definition: String definition to be added.
        """
        self.definition = definition

    @abstractmethod
    def get_query(self):
        pass


class SpecificConstraint(Constraint):
    """
    Represents a specific constraint with proper noun.
    """
    def __init__(self, description, is_verifiable,):
        super().__init__(description, is_verifiable)
    
    def get_query(self) -> list[str]:
        """
        Retrieves the query for a specific constraint.
        
        :return: List containing the description of the constraint.
        """
        return [self.description]


class GeneralConstraint(Constraint):
    """
    Represents a general constraint that may need expansions.
    """
    def __init__(self, description, is_verifiable):
        super().__init__(description, is_verifiable)
        self.expansion = [self.description]

    def get_query(self) -> list[str]:
        """
        Retrieves the expanded queries for a general constraint.
        
        :return: List of expanded descriptions.
        """
        return self.expansion

    def expand(self, new_description: str):
        """
        Expands the general constraint by adding new descriptions.
        
        :param new_description: String description to add to the expansions.
        """
        self.expansion.append(new_description)


class Query(ABC):
    def __init__(self, description: str):
        self.description= description
    
    @abstractmethod
    def get_descriptions(self) -> list[str]:
        pass

    @abstractmethod
    def get_description_weights(self) -> list[float]:
        pass

    @abstractmethod
    def get_specific_constraints(self) -> list[str]:
        pass
    
class QueryQE(Query):
    """
    Query expansion model that handles query descriptions and allows for adding expansions.
    """
    def __init__(self, description: str):
        super().__init__(description)
        self.expansions = []
    
    def add_expansion(self, expansion: str):
        """
        Adds an expansion to the query.
        
        :param expansion: String representing the expansion to be added.
        """
        self.expansions.append(expansion)
    
    def get_descriptions(self) -> list[str]:
        """
        Retrieves all expansions as query descriptions.
        
        :return: List of expansions.
        """
        return self.expansions
    
    def get_description_weights(self) -> list[float]:
        """
        Retrieves uniform weights for each expansion.
        
        :return: List of equal weights for each description.
        """
        return [1 for _ in self.expansions]
    
    def get_specific_constraints(self) -> list[str]:
        """
        Retrieves all specific constraints from the query.
        
        :return: Empty list since query expansion does not contain specific constraints.
        """
        return []

class QueryCE(Query):
    """
    Constraints expansion model that supports adding constraints.
    """
    def __init__(self, description: str):
        super().__init__(description)
        self.constraints = []

    def add_constraint(self, constraint: Constraint):
        """
        Adds a constraint to the query.
        
        :param constraint: Constraint object to be added.
        """
        self.constraints.append(constraint)

    def get_descriptions(self) -> list[str]:
        """
        Compiles descriptions from all constraints.
        
        :return: List of constraint descriptions.
        """
        descriptions = [c.definition for c in self.constraints]
        return descriptions
    
    def get_description_weights(self) -> list[float]:
        """
        Calculates and returns weights for each description based on its source constraint.
        
        :return: List of weights for descriptions.
        """
        # weights = []
        # for c in self.constraints:
        #     if isinstance(c, SpecificConstraint):
        #         w = [0.5]
        #     else:
        #         ds = c.get_query()
        #         w = [1 / len(ds) for _ in ds]
        #     weights.extend(w)
        # return weights
        return [1 for _ in self.constraints]
    
    def get_specific_constraints(self) -> list[str]:
        """
        Retrieves all specific constraints from the query.
        
        :return: List of specific constraints.
        """
        return [c.description for c in self.constraints if isinstance(c, SpecificConstraint)]
        