require 'octane'
require 'minitest/autorun'
include Octane

EPOCHS=2

class TestBackprop < MiniTest::Unit::TestCase

	def setup
		@single_layer=Network.new 
		@single_layer.add_layer(2) # two inputs
		@single_layer.add_layer(4) # five hidden neurons
		@single_layer.add_layer(1) # one output

		@multi_layer=Network.new 
		@multi_layer.add_layer(2) # two inputs
		@multi_layer.add_layer(4) # five hidden neurons
		@multi_layer.add_layer(4) # five hidden neurons
		@multi_layer.add_layer(1) # one output

		function = ->(x,y) { Math.cos(x+y) } # let's try to approximate cosine of sum of two variables 

		@test_dataset=Dataset.new
		@train_dataset=Dataset.new

		# create two datasets, 100 and 10000 samples respectively
		[@test_dataset, @train_dataset].zip([100, 1000]).each do |dataset, samples| 
		    samples.times do 
		        input=[rand, rand]
		        dataset.add_sample(input, function[*input]) 
		    end
		end
	end

	def test_batch
		@single_layer.train_one(@train_dataset.sample)
	end

	def test_that_network_learns
		before_training=@single_layer.test(@test_dataset)
		@single_layer.train(@train_dataset)
		after_training=@single_layer.test(@test_dataset)

		puts "Single layer: Before: #{before_training}, after: #{after_training}"
		assert_operator before_training, :> , after_training

		before_training=@multi_layer.test(@test_dataset)
		@multi_layer.train(@train_dataset)
		after_training=@multi_layer.test(@test_dataset)
		
		puts "Multi-layered: Before: #{before_training}, after: #{after_training}"
		assert_operator before_training, :> , after_training

	end

	def test_disabling_units
		@single_layer.hidden_layer.disable(0)
		assert @single_layer.hidden_layer.unit(0).disabled

		before_training=@single_layer.hidden_layer.unit(0).input_weights
		@single_layer.train(@train_dataset,epochs: EPOCHS)
		after_training=@single_layer.hidden_layer.unit(0).input_weights

		assert_equal before_training, after_training

		@single_layer.hidden_layer.enable(0)
		refute @single_layer.hidden_layer.unit(0).disabled

		before_training=@single_layer.hidden_layer.unit(0).input_weights
		@single_layer.train(@train_dataset, epochs: EPOCHS)
		after_training=@single_layer.hidden_layer.unit(0).input_weights

		refute_equal before_training, after_training
	end

	def test_that_dropout_works
		no_dropout_nn=Network.new 0.1
		no_dropout_nn.add_layer(2)
		no_dropout_nn.add_layer(10)
		no_dropout_nn.add_layer(1)

		dropout_nn=no_dropout_nn.copy
		

		1000.times do
			no_dropout_nn.hidden_layer.size.times do |neuron|
				if rand>0.5
					dropout_nn.hidden_layer.disable(neuron)
				end
			end

			sample=@train_dataset.sample
			
			dropout_nn.train_one(sample)
			no_dropout_nn.train_one(sample)

			dropout_nn.hidden_layer.enable_all
		end

		
		dropout_nn.output_layer.each {|unit| unit.input_weights.map! {|w| w/2}; unit.input_weights[unit.input_weights.length-1]*=2;}

		puts "Without dropout: #{no_dropout_nn.test(@test_dataset)}"
		puts "With dropout: #{dropout_nn.test(@test_dataset)}"
	end

	def test_batch_learning

		
		@net=Network.new(0.1)
		@net.add_layer(2)
		@net.add_layer(4)
		@net.add_layer(1)


		before_training=@net.test(@test_dataset)
		
		100.times do
			sample=@train_dataset.sample 10
			@net.train_batch(sample)
		end

		after_training=@net.test(@test_dataset)

		puts "Batch learning: Before: #{before_training}, after: #{after_training}"
		assert_operator before_training, :>, after_training


	end
end