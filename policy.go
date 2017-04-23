package cuberl

import (
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyrl"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/gocube"
)

// NewPolicy creates a new policy RNN.
func NewPolicy(c anyvec.Creator, hidden, layers int) anyrnn.Block {
	inScale := c.MakeNumeric(0x10)
	res := anyrnn.Stack{
		anyrnn.NewLSTM(c, CubeVectorSize, hidden).ScaleInWeights(inScale),
	}
	for i := 1; i < layers; i++ {
		res = append(res, anyrnn.NewLSTM(c, hidden, hidden))
	}
	res = append(res, &anyrnn.LayerBlock{
		Layer: anynet.Net{
			anynet.NewFC(c, hidden, NumActions),
		},
	})
	return res
}

// PolicyMoves runs the policy on the start state to get
// a sequence of n moves.
// If greedy is false, then moves are sampled rather than
// being taken greedily.
func PolicyMoves(a anyrnn.Block, s *State, n int, greedy bool) []gocube.Move {
	cr := policyCreator(a)
	bs := a.Start(1)
	res := []gocube.Move{}
	for i := 0; i < n; i++ {
		out := a.Step(bs, CubeVector(cr, &s.Cube))
		bs = out.State()

		var move gocube.Move
		if greedy {
			move = gocube.Move(anyvec.MaxIndex(out.Output()))
		} else {
			sampled := anyrl.Softmax{}.Sample(out.Output(), 1)
			move = gocube.Move(anyvec.MaxIndex(sampled))
		}
		s, _ = s.Move(move)
		res = append(res, move)
	}
	return res
}

func policyCreator(b anyrnn.Block) anyvec.Creator {
	return b.(anynet.Parameterizer).Parameters()[0].Vector.Creator()
}
