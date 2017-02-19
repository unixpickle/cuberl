package cuberl

import (
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/gocube"
)

// NewAgent creates a new agent RNN.
func NewAgent(c anyvec.Creator, hidden int) anyrnn.Block {
	inScale := c.MakeNumeric(0x10)
	return anyrnn.Stack{
		anyrnn.NewLSTM(c, CubeVectorSize, hidden).ScaleInWeights(inScale),
		anyrnn.NewLSTM(c, hidden, hidden),
		&anyrnn.LayerBlock{
			Layer: anynet.Net{
				anynet.NewFC(c, hidden, NumActions),
				anynet.LogSoftmax,
			},
		},
	}
}

// AgentMoves runs the agent on the start state to get
// a sequence of n moves.
// If greedy is false, then moves are sampled rather than
// being taken greedily.
func AgentMoves(a anyrnn.Block, s *State, n int, greedy bool) []gocube.Move {
	cr := agentCreator(a)
	bs := a.Start(1)
	res := []gocube.Move{}
	for i := 0; i < n; i++ {
		out := a.Step(bs, CubeVector(cr, &s.Cube))
		bs = out.State()

		var move gocube.Move
		if greedy {
			move = gocube.Move(anyvec.MaxIndex(out.Output()))
		} else {
			move = gocube.Move(sampleMove(out.Output().Copy()))
		}
		s, _ = s.Move(move)
		res = append(res, move)
	}
	return res
}

func agentCreator(b anyrnn.Block) anyvec.Creator {
	return b.(anynet.Parameterizer).Parameters()[0].Vector.Creator()
}
