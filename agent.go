package cuberl

import (
	"math/rand"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/gocube"
)

// An Agent is a stateful agent.
type Agent struct {
	Block anyrnn.Block
}

// QSamples creates sequence-to-sequences samples that
// adjust the agent's value function using Q-Learning.
func (a *Agent) QSamples(start []*State, steps int, discount, explore float64) *anys2s.Batch {
	state := append([]*State{}, start...)
	blockState := a.Block.Start(len(start))
	cr := a.creator()

	var ins, outs []*anyseq.Batch
	present := make([]bool, len(start))
	for i := range present {
		present[i] = true
	}

	var lastOut anyvec.Vector
	var lastReward []float64
	for i := 0; i < steps; i++ {
		cubeIns := make([]anyvec.Vector, len(state))
		for i, s := range state {
			cubeIns[i] = CubeVector(cr, &s.Cube)
		}
		inVec := cr.Concat(cubeIns...)
		ins = append(ins, &anyseq.Batch{Packed: inVec, Present: present})

		res := a.Block.Step(blockState, inVec)
		blockState = res.State()

		if i > 0 {
			discounted := res.Output().Copy()
			discounted.Scale(cr.MakeNumeric(discount))
			outs = append(outs, &anyseq.Batch{
				Packed:  correctedPred(lastOut, discounted, lastReward),
				Present: present,
			})
		}

		lastOut = res.Output()
		lastReward := make([]float64, len(state))
		for i, s := range state {
			moveIdx := anyvec.MaxIndex(lastOut.Slice(i*NumActions, (i+1)*NumActions))
			if rand.Float64() < explore {
				moveIdx = rand.Intn(NumActions)
			}
			state[i], lastReward[i] = s.Move(gocube.Move(moveIdx))
		}
	}

	// No correction for the last output.
	outs = append(outs, &anyseq.Batch{Packed: lastOut, Present: present})

	return &anys2s.Batch{
		Inputs:  anyseq.ConstSeq(cr, ins),
		Outputs: anyseq.ConstSeq(cr, outs),
	}
}

func (a *Agent) creator() anyvec.Creator {
	return a.Block.(anynet.Parameterizer).Parameters()[0].Vector.Creator()
}

func correctedPred(predictions, nextOut anyvec.Vector, immRew []float64) anyvec.Vector {
	c := predictions.Creator()
	n := predictions.Len() / NumActions
	maxMap := anyvec.MapMax(nextOut, NumActions)
	greedyValues := c.MakeVector(n)
	maxMap.Map(nextOut, greedyValues)
	greedyValues.Add(c.MakeVectorData(c.MakeNumericList(immRew)))

	differences := c.MakeVector(n)
	maxMap.Map(predictions, differences)
	differences.Scale(c.MakeNumeric(-1))
	differences.Add(greedyValues)

	// Use MapTranspose to add the differences.
	res := predictions.Copy()
	maxMap.MapTranspose(differences, res)
	return res
}
