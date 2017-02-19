package cuberl

import (
	"math/rand"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anynet/anys2s"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/gocube"
)

// QSamples creates sequence-to-sequences samples that
// adjust the agent's value function using Q-Learning.
func QSamples(agent anyrnn.Block, start []*State, steps int, discount,
	explore float64) *anys2s.Batch {
	state := append([]*State{}, start...)
	blockState := agent.Start(len(start))
	cr := agentCreator(agent)

	var ins, outs []*anyseq.Batch
	present := make([]bool, len(start))
	for i := range present {
		present[i] = true
	}

	var lastOut anyvec.Vector
	var lastReward []float64
	var lastMoves []int
	for i := 0; i < steps; i++ {
		cubeIns := make([]anyvec.Vector, len(state))
		for j, s := range state {
			cubeIns[j] = CubeVector(cr, &s.Cube)
		}
		inVec := cr.Concat(cubeIns...)
		ins = append(ins, &anyseq.Batch{Packed: inVec, Present: present})

		res := agent.Step(blockState, inVec)
		blockState = res.State()

		if i > 0 {
			discounted := res.Output().Copy()
			discounted.Scale(cr.MakeNumeric(discount))
			outs = append(outs, &anyseq.Batch{
				Packed:  correctedPred(lastOut, discounted, lastMoves, lastReward),
				Present: present,
			})
		}

		lastOut = res.Output()
		lastReward = make([]float64, len(state))
		lastMoves = make([]int, len(state))
		for j, s := range state {
			moveIdx := anyvec.MaxIndex(lastOut.Slice(j*NumActions, (j+1)*NumActions))
			if rand.Float64() < explore {
				moveIdx = rand.Intn(NumActions)
			}
			lastMoves[j] = moveIdx
			state[j], lastReward[j] = s.Move(gocube.Move(moveIdx))
		}
	}

	// No correction for the last output.
	outs = append(outs, &anyseq.Batch{Packed: lastOut, Present: present})

	return &anys2s.Batch{
		Inputs:  anyseq.ConstSeq(cr, ins),
		Outputs: anyseq.ConstSeq(cr, outs),
	}
}

func correctedPred(predictions, nextOut anyvec.Vector, moves []int,
	immRew []float64) anyvec.Vector {
	c := predictions.Creator()
	n := predictions.Len() / NumActions
	maxMap := anyvec.MapMax(nextOut, NumActions)
	greedyValues := c.MakeVector(n)
	maxMap.Map(nextOut, greedyValues)
	greedyValues.Add(c.MakeVectorData(c.MakeNumericList(immRew)))

	movesMapIdx := make([]int, len(moves))
	for i, m := range moves {
		movesMapIdx[i] = m + i*NumActions
	}
	movesMap := c.MakeMapper(predictions.Len(), movesMapIdx)

	oldGuess := c.MakeVector(n)
	movesMap.Map(predictions, oldGuess)
	greedyValues.Sub(oldGuess)

	// Use MapTranspose to add the differences.
	res := predictions.Copy()
	movesMap.MapTranspose(greedyValues, res)
	return res
}
