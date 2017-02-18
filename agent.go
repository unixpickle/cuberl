package cuberl

import (
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
)

// NewAgent creates a new agent RNN.
func NewAgent(c anyvec.Creator, hidden int) anyrnn.Block {
	inScale := c.MakeNumeric(6)
	return anyrnn.Stack{
		anyrnn.NewLSTM(c, CubeVectorSize, hidden).ScaleInWeights(inScale),
		anyrnn.NewLSTM(c, hidden, hidden),
		&anyrnn.LayerBlock{
			Layer: anynet.Net{
				anynet.NewFC(c, hidden, NumActions),
			},
		},
	}
}

func agentCreator(b anyrnn.Block) anyvec.Creator {
	return b.(anynet.Parameterizer).Parameters()[0].Vector.Creator()
}
