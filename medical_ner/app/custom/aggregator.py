from nvflare.app_common.abstract.aggregator import Aggregator

## check this for example
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator



class myAggregator(Aggregator):
    @abstractmethod
    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        """Accept the shareable submitted by the client.

        Args:
            shareable: submitted Shareable object
            fl_ctx: FLContext

        Returns:
            first boolean to indicate if the contribution has been accepted.

        """
        pass

    @abstractmethod
    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        """Perform the aggregation for all the received Shareable from the clients.

        Args:
            fl_ctx: FLContext

        Returns:
            shareable
        """
        pass