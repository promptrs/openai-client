#[allow(warnings)]
mod bindings;

use a2httpc::Error;
use a2httpc::body::Json;
use a2httpc::header::CONTENT_TYPE;
use a2httpc::{ResponseReader, TextReader};
use either::IntoEither;
use log::warn;
use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize, Serializer};
use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Lines, Write};
use std::iter::StepBy;
use std::time::Duration;

use bindings::promptrs::core::types;
use bindings::{CompletionRequest, Guest};

#[derive(Serialize)]
struct Compl<'p>(#[serde(with = "CompletionParams")] &'p types::CompletionParams);

#[derive(Serialize)]
#[serde(remote = "types::CompletionParams")]
struct CompletionParams {
	model: String,
	temperature: Option<f64>,
	top_p: Option<f64>,
	#[serde(serialize_with = "serialize_messages")]
	messages: Vec<types::Message>,
	stream: bool,
}

fn serialize_messages<S: Serializer>(
	messages: &Vec<types::Message>,
	serializer: S,
) -> Result<S::Ok, S::Error> {
	let mut seq = serializer.serialize_seq(Some(messages.len()))?;

	let mut messages = messages.iter();
	let mut curr = messages.next();

	let (mut abuf, mut tbuf) = (String::new(), String::new());

	loop {
		(abuf.truncate(0), tbuf.truncate(0));
		while let Some(
			types::Message::ToolCall((assistant, tool)) | types::Message::Status((assistant, tool)),
		) = curr
		{
			abuf.push_str(&assistant);
			tbuf.push_str(&tool);
			curr = messages.next();
		}
		seq.serialize_element(&HashMap::from([("role", "assistant"), ("content", &abuf)]))?;
		seq.serialize_element(&HashMap::from([("role", "tool"), ("content", &tbuf)]))?;

		let Some(message) = curr else { break };

		let (role, content) = match message {
			types::Message::System(content) => ("system", content),
			types::Message::User(content) => ("user", content),
			types::Message::Assistant(content) => ("assistant", content),
			_ => unreachable!(),
		};
		let map = HashMap::from([("role", role), ("content", content)]);
		seq.serialize_element(&map)?;

		curr = messages.next();
	}
	seq.end()
}

struct Component;

impl Guest for Component {
	fn completion(payload: CompletionRequest) -> Result<String, String> {
		payload.chat_completion().map_err(|err| err.to_string())
	}
}

impl CompletionRequest {
	pub fn chat_completion(&self) -> Result<String, Error> {
		let response = self
			.stream()?
			.into_iter()
			.fold(String::new(), |acc, chunk| {
				let Ok(chunk) = chunk else {
					warn!("{:?}", chunk);
					return acc;
				};
				let text = chunk
					.choices
					.into_iter()
					.filter_map(|c| c.delta.content)
					.fold("".to_string(), |acc, s| acc + s.as_str());
				print!("{}", text);
				_ = io::stdout().flush();
				acc + text.as_str()
			});
		println!("\n----------END_OF_RESPONSE----------\n\n");

		Ok(response)
	}

	fn stream(&self) -> Result<ChatCompletionStream, Error> {
		let (status, _, reader) = a2httpc::post(self.base_url.to_string() + "/v1/chat/completions")
			.read_timeout(Duration::from_secs(600))
			.into_either(self.api_key.is_some())
			.right_or_else(|req| req.bearer_auth(self.api_key.as_ref().as_slice()[0]))
			.header(CONTENT_TYPE, "application/json")
			.body(Json(Compl(&self.body)))
			.send()
			.inspect_err(|_| println!("here"))?
			.split();
		if !status.is_success() {
			let resp = reader.text()?;
			return Err(Error::from(io::Error::new(
				io::ErrorKind::ConnectionRefused,
				format!("Status Code: {status}\nResponse: {resp}"),
			)));
		}

		Ok(ChatCompletionStream(
			BufReader::new(TextReader::new(reader, a2httpc::charsets::UTF_8))
				.lines()
				.step_by(2),
		))
	}
}

pub struct ChatCompletionStream(StepBy<Lines<BufReader<TextReader<ResponseReader>>>>);

impl Iterator for ChatCompletionStream {
	type Item = Result<ChatCompletionChunk, io::Error>;

	fn next(&mut self) -> Option<Self::Item> {
		self.0
			.next()
			.and_then(|s| s.and_then(next_chunk).transpose())
	}
}

fn next_chunk(mut s: String) -> Result<Option<ChatCompletionChunk>, io::Error> {
	if s.find("[DONE]").is_some() {
		return Ok(None);
	}
	if let Some(i) = s.find('{') {
		s = s.split_off(i);
	}
	serde_json::from_str(&s)
		.map_err(|err| io::Error::new(io::ErrorKind::InvalidData, format!("Malformed JSON: {err}")))
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunk {
	pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
	#[serde(alias = "message")]
	pub delta: Delta,
}

#[derive(Debug, Deserialize)]
pub struct Delta {
	pub content: Option<String>,
}

bindings::export!(Component with_types_in bindings);
